"""对照版 job：``POST /api/task/{id}/bilingual`` → ``bi/`` 回填 + 编译 + ``bi_pdf`` 登记。"""

from __future__ import annotations

import json
import sqlite3
import time
from functools import partial
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from conftest import FakeEngine, make_app, mk_task_row, upload_tex, wait_terminal
from starlette.testclient import TestClient

from texlate.server.store import Store
from texlate.server.worker.bilingual import _bi_text, _brace_depths
from texlate.xlat.pipeline import MockTranslator

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class TestBraceDepths:
    def test_nesting_comments_and_escapes(self) -> None:
        text = "a {b {c} \\{ % {\nd } e"
        got = _brace_depths(text, [0, 3, 6, text.index("d"), len(text) - 1])
        assert got[0] == 0
        assert got[3] == 1  # 进 {b
        assert got[6] == 2  # noqa: PLR2004 -- 进 {c
        # {c} 已闭合、\{ 不计、% 注释内 { 不计 → d 处仍在 {b 内
        assert got[text.index("d")] == 1
        assert got[len(text) - 1] == 0


def test_bi_text_puts_original_first_then_translation() -> None:
    out = _bi_text("Hello [[MATH_1]].", "你好 [[MATH_1]]。")
    assert (
        out
        == "\\texlateorigbegin Hello [[MATH_1]].\\texlateorigend\n\n你好 [[MATH_1]]。"
    )


def _wait_event(db: Path, tid: str, needle: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        con = sqlite3.connect(db)
        try:
            rows = con.execute(
                "SELECT data FROM task_events WHERE task_id=?", (tid,)
            ).fetchall()
        finally:
            con.close()
        if any(needle in json.loads(r[0]).get("line", "") for r in rows):
            return
        time.sleep(0.1)
    msg = f"event {needle!r} not seen for {tid}"
    raise AssertionError(msg)


class TestBilingualJob:
    @pytest.fixture
    def bi_client(
        self,
        tmp_path: Path,
        clean_env: pytest.MonkeyPatch,  # noqa: ARG002 -- fixture 副作用
    ) -> Iterator[tuple[TestClient, Path]]:
        engine = FakeEngine()
        app = make_app(
            tmp_path,
            start_worker=True,
            translator_factory=lambda _ctx: MockTranslator(),
            engine_factory=lambda _name: engine,
        )
        with TestClient(app) as c:
            yield c, tmp_path / "data" / "texlate.db"

    def test_builds_and_registers_bi_pdf(
        self, bi_client: tuple[TestClient, Path]
    ) -> None:
        c, db = bi_client
        tid = upload_tex(c)["task_id"]
        assert wait_terminal(c, tid)["status"] == "done"
        r = c.post(f"/api/task/{tid}/bilingual")
        assert r.status_code == HTTPStatus.ACCEPTED
        assert r.json() == {"task_id": tid, "status": "queued"}
        _wait_event(db, tid, "bilingual: bi.pdf 已生成")
        s = Store(db)
        s.open()
        try:
            row = s.get(tid)
            files = s.files(tid)
        finally:
            s.close()
        assert row is not None
        assert row["status"] == "done"  # 旁路 job 不迁终态
        assert "bi_pdf" in files
        root = c.app.state.data_dir / "tasks" / tid / "bi"
        tex = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.tex"))
        assert "\\newcommand{\\texlateorigbegin}" in tex
        assert "\\texlateorigbegin " in tex
        assert c.get(f"/api/files/{tid}/bi.pdf").status_code == HTTPStatus.OK

    def test_route_rejects_unknown_and_active(
        self, bi_client: tuple[TestClient, Path]
    ) -> None:
        c, _db = bi_client
        assert c.post("/api/task/t_nope/bilingual").status_code == HTTPStatus.NOT_FOUND
        tid = c.portal.call(partial(mk_task_row, c.app.state.store))["id"]
        r = c.post(f"/api/task/{tid}/bilingual")
        assert r.status_code == HTTPStatus.CONFLICT
        assert r.json()["code"] == "invalid_state"

    def test_enqueue_requires_source_tree(
        self, bi_client: tuple[TestClient, Path]
    ) -> None:
        c, _db = bi_client
        store = c.app.state.store
        runner = c.app.state.runner
        tid = c.portal.call(partial(mk_task_row, store))["id"]
        c.portal.call(partial(store.transition, tid, "done", force=True))
        with pytest.raises(ValueError, match="materialized source tree"):
            c.portal.call(partial(runner.enqueue_bilingual, tid))
