"""整篇重译创建独立任务，并隔离原任务与持久段缓存。"""

from __future__ import annotations

import json
from functools import partial
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from conftest import force_status, get_row, mk_api_task, reg_artifact

from texlate.server.store import ACTIVE_STATUSES, TERMINAL_STATUSES
from texlate.server.worker import Secrets, TaskCtx
from texlate.server.worker.translate.cache import _NullCache

if TYPE_CHECKING:
    from starlette.testclient import TestClient

ARXIV = "2401.00001v2"
KEY = {"X-Texlate-Key": "sk-test"}


def test_retranslate_always_creates_and_preserves_original(client: TestClient) -> None:
    tid = mk_api_task(
        client,
        ARXIV,
        headers={**KEY, "Idempotency-Key": "original"},
        model="old-model",
        target_lang="zh-TW",
        options={
            "main": "paper.tex",
            "front_matter": {"title": True},
            "auto_glossary": False,
            "concurrency": 2,
            "glossary": "terms",
        },
    )
    force_status(client, tid, "done")
    reg_artifact(client, tid, "zh_pdf", "zh.pdf", b"paid translation")
    store = client.app.state.store
    opts = json.loads(get_row(client, tid)["options_json"])
    opts.update(
        {
            "mock_run": 1,
            "reuse_hit": "old",
            "engine_resolved": "xelatex",
            "route_engines": ["xelatex"],
            "arxiv_categories": ["cs.AI"],
            "served_by": {"old": 10},
            "reasoning_effort": "high",
        }
    )
    client.portal.call(partial(store.update_fields, tid, options_json=json.dumps(opts)))
    before = get_row(client, tid)
    files_before = client.portal.call(partial(store.files, tid))
    # 当前默认模型须覆盖原模型；相同幂等头不得复用原行或第一次重译行。
    settings = client.put(
        "/api/channels",
        json={
            "channels": [
                {
                    "id": "current",
                    "name": "Current",
                    "preset": "custom",
                    "base_url": "https://api.deepseek.com",
                    "protocol": "auto",
                    "models": [{"model": "current-model", "redirect_model": ""}],
                    "api_key": "sk-current",
                    "enabled": True,
                }
            ],
            "route": {"channel_id": "current", "model": "current-model"},
        },
    )
    assert settings.status_code == HTTPStatus.OK, settings.text
    results = [
        client.post(
            f"/api/task/{tid}/retranslate",
            json={},
            headers={"Idempotency-Key": "original"},
        )
        for _ in range(2)
    ]
    assert [r.status_code for r in results] == [HTTPStatus.ACCEPTED] * 2
    ids = [r.json()["task_id"] for r in results]
    assert tid not in ids
    assert ids[0] != ids[1]
    for new_id in ids:
        row = get_row(client, new_id)
        assert row["model"] == "current-model"
        assert row["target_lang"] == "zh-TW"
        assert row["source_name"] == ARXIV
        assert row["kind"] == "arxiv"
        new_opts = json.loads(row["options_json"])
        assert new_opts == {
            "source": "eprint",
            "main": "paper.tex",
            "front_matter": {"title": True},
            "auto_glossary": False,
            "concurrency": 2,
            "glossary": "terms",
            "prefer": "fresh",
            "no_seg_cache": True,
        }
    assert get_row(client, tid) == before
    assert client.portal.call(partial(store.files, tid)) == files_before
    assert (
        client.app.state.data_dir / "tasks" / tid / "zh.pdf"
    ).read_bytes() == b"paid translation"
    assert client.get(f"/api/task/{tid}").status_code == HTTPStatus.OK


def test_resolved_version_overrides_original_unversioned_request(
    client: TestClient,
) -> None:
    tid = mk_api_task(client, "2610.31668", headers=KEY)
    client.portal.call(
        partial(
            client.app.state.store.update_fields,
            tid,
            arxiv_id="2610.31668v2",
            main_tex="auto-selected.tex",
        )
    )
    force_status(client, tid, "done")
    response = client.post(f"/api/task/{tid}/retranslate", json={}, headers=KEY)
    assert response.status_code == HTTPStatus.ACCEPTED, response.text
    row = get_row(client, response.json()["task_id"])
    assert row["source_name"] == "2610.31668v2"
    assert "main" not in json.loads(row["options_json"])


@pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
def test_terminal_states_create_new_task(client: TestClient, status: str) -> None:
    tid = mk_api_task(client, ARXIV, headers=KEY)
    force_status(client, tid, status)
    response = client.post(f"/api/task/{tid}/retranslate", json={}, headers=KEY)
    assert response.status_code == HTTPStatus.ACCEPTED, response.text
    assert response.json()["task_id"] != tid
    assert get_row(client, tid)["status"] == status


@pytest.mark.parametrize("status", sorted(ACTIVE_STATUSES))
def test_active_original_rejected(client: TestClient, status: str) -> None:
    tid = mk_api_task(client, ARXIV, headers=KEY)
    force_status(client, tid, status)
    before = get_row(client, tid)
    response = client.post(f"/api/task/{tid}/retranslate", json={}, headers=KEY)
    assert response.status_code == HTTPStatus.CONFLICT
    assert get_row(client, tid) == before


@pytest.mark.parametrize("body", [{"model": "x"}, {"options": {}}, [], None, "x"])
def test_invalid_body_rejected(client: TestClient, body: object) -> None:
    tid = mk_api_task(client, ARXIV)
    response = client.post(
        f"/api/task/{tid}/retranslate",
        content=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST


def test_unsupported_kind_rejected(client: TestClient) -> None:
    tid = mk_api_task(client, ARXIV)
    client.portal.call(
        partial(client.app.state.store.update_fields, tid, kind="upload_tex")
    )
    response = client.post(f"/api/task/{tid}/retranslate", json={})
    assert response.status_code == HTTPStatus.BAD_REQUEST


def test_html_source_preserved(client: TestClient) -> None:
    tid = mk_api_task(client, ARXIV, options={"source": "html"})
    response = client.post(f"/api/task/{tid}/retranslate", json={})
    assert response.status_code == HTTPStatus.ACCEPTED
    row = get_row(client, response.json()["task_id"])
    assert row["kind"] == "arxiv_html"
    assert json.loads(row["options_json"])["source"] == "html"
    assert row["status"] == "needs_auth"


def test_task_tenant_guard(server_client: TestClient) -> None:
    tid = mk_api_task(server_client, ARXIV, headers=KEY)
    response = server_client.post(
        f"/api/task/{tid}/retranslate", json={}, headers={"X-Texlate-Key": "sk-other"}
    )
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert (
        server_client.post(
            "/api/task/t_missing/retranslate", json={}, headers=KEY
        ).status_code
        == HTTPStatus.NOT_FOUND
    )


def test_retranslate_cache_has_no_persistent_reads_or_writes(
    client: TestClient,
) -> None:
    tid = mk_api_task(client, ARXIV)
    response = client.post(f"/api/task/{tid}/retranslate", json={}, headers=KEY)
    row = get_row(client, response.json()["task_id"])
    store = client.app.state.store
    client.portal.call(
        partial(store.cache_put_batch, [("existing", "old translation", "m", "zh-CN")])
    )
    ctx = TaskCtx(
        task_id=row["id"],
        root=client.app.state.data_dir / "tasks" / row["id"],
        row=row,
        secrets=Secrets(api_key="real-key"),
        store=client.app.state.store,
        bus=client.app.state.bus,
    )
    cache = client.app.state.runner.worker._make_cache(ctx)  # noqa: SLF001 -- 验证内部持久缓存闸
    assert isinstance(cache, _NullCache)
    cache.prewarm(["existing"])
    assert "existing" not in cache
    with pytest.raises(KeyError):
        _ = cache["existing"]
    cache["new"] = "fresh translation"
    assert cache.drain() == []
    assert cache["new"] == "fresh translation"
    assert cache.drain() == []
    assert client.portal.call(partial(store.cache_get, "existing")) == "old translation"
    assert client.portal.call(partial(store.cache_get, "new")) is None
