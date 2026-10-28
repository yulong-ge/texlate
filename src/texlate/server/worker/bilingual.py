r"""``PipelineWorker._Bilingual``——终态任务的中英对照 PDF（``bi.pdf``）按需生成 job。

``runner.enqueue_bilingual`` 入队、dispatcher 与主任务同一串行域派发。只读已落库
译文，不发翻译请求：源树（``base/``）按块回填——顶层正文段（``_BI_ENVS`` 环境内且
不在命令参数里）写成「灰色原文 + 译文」，其余块只放译文——再注入中文支持与原文样式宏
后单独编译。
任务本体**不迁终态**；失败只记 log/warning，``zh.pdf`` 等既有产物不受影响。

原文副本内 ``\label`` 置空、公式/脚注计数器在副本前后存取复原——引用与编号
只认译文那份，不因原文重复出现而错位。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from collections import Counter
from typing import TYPE_CHECKING

from texlate.compile._docseams import _splice_before_document
from texlate.compile.inject import prepare_chinese
from texlate.latex.reconstruct import reconstruct
from texlate.pipecore import compile_judge

from ._common import TaskCtx, chunk_db_id
from .compile.splice import _delivered_map

if TYPE_CHECKING:
    from texlate.server.events import EventBus
    from texlate.server.store import Store

log = logging.getLogger(__name__)

#: 写成「原文 + 译文」的块：正文段（``context == "para"``）且所在环境在此集内。
#: 标题/图注/条目等短块与盒子/表格等环境内的段只放译文——原文副本带 ``\par``
#: 与分组，进 tcolorbox/minipage/表格单元会破坏环境配对（2610.20331 实案）
_BI_ENVS = frozenset({None, "abstract"})

#: 原文副本样式与计数器保护（preamble 尾注入）。color 包未载才补 xcolor——
#: 已载带选项时再载会 option clash
_BI_PREAMBLE = r"""\makeatletter
\@ifpackageloaded{xcolor}{}{\@ifpackageloaded{color}{}{\RequirePackage{xcolor}}}
\newcount\texlate@eqsave
\newcount\texlate@fnsave
\newcommand{\texlateorigbegin}{\par\begingroup
  \@ifundefined{c@equation}{}{\global\texlate@eqsave=\value{equation}}%
  \@ifundefined{c@footnote}{}{\global\texlate@fnsave=\value{footnote}}%
  \let\label\@gobble\color{gray}}
\newcommand{\texlateorigend}{\par\endgroup
  \@ifundefined{c@equation}{}{\setcounter{equation}{\texlate@eqsave}}%
  \@ifundefined{c@footnote}{}{\setcounter{footnote}{\texlate@fnsave}}}
\makeatother
"""


class BilingualBuildError(RuntimeError):
    r"""对照版无法产出（无 ``\begin{document}`` 锚 / 编译无 pdf）。"""


def _brace_depths(text: str, positions: list[int]) -> dict[int, int]:
    r"""各位置处的花括号嵌套深度（跳过 ``%`` 注释与反斜杠转义）。

    深度 > 0 = 落在某命令参数内（如 ``\mybox{...}``）——段落的 env 探测看不到这类
    包裹，原文副本的 ``\par``/分组进去会破坏 tcolorbox 等盒子（2610.20331 实案）。
    """
    out: dict[int, int] = {}
    depth = 0
    i = 0
    n = len(text)
    for t in sorted(set(positions)):
        while i < t:
            ch = text[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "%":
                j = text.find("\n", i)
                i = n if j < 0 else j
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        out[t] = depth
    return out


def _bi_text(src: str, zh: str) -> str:
    """正文块对照形：原文（灰、计数器隔离）在前，译文在后，各成段。"""
    return f"\\texlateorigbegin {src}\\texlateorigend\n\n{zh}"


class _Bilingual:
    """对照版 job mixin：回填 base/ → 注入 → 编译 → 登记 ``bi_pdf``。"""

    if TYPE_CHECKING:
        bus: EventBus
        store: Store
        _compile_timeout: float

    async def run_bilingual(self, ctx: TaskCtx) -> None:
        """``TaskRunner._bilingual_job`` 派发入口：loop 装配 → job 体 → 线程排空。"""
        self._rehydrate_ctx(ctx)
        try:
            await self._ensure_scans(ctx)
            ctx.expect_cjk = self._expect_cjk(ctx)
            rows = self._all_chunks(ctx)
            # env 可译性判定已在主编译时落库（判否块为 fallback_orig，不在交付映射内），
            # 此处不重跑——重跑会再发 LLM 请求并改写 chunks 行
            trans = _delivered_map(rows)
            await self._to_thread(ctx, self._bi_build, trans)
            self._log(ctx, "bilingual: bi.pdf 已生成", force=True)
        except asyncio.CancelledError:
            ctx.cancel_flag.set()
            raise
        except Exception as e:
            log.exception("bilingual job failed for %s", ctx.task_id)
            with contextlib.suppress(Exception):
                self._warning(
                    ctx,
                    "bilingual",
                    f"对照版生成失败: {type(e).__name__}: {e}",
                    force=True,
                )
        finally:
            await self._drain_threads(ctx)

    def _bi_build(self, ctx: TaskCtx, trans: dict[str, str]) -> None:
        """Worker 线程段：bi/ 工程物化 + 编译 + 登记。"""
        bi_dir = ctx.root / "bi"
        if bi_dir.exists():
            shutil.rmtree(bi_dir)
        shutil.copytree(ctx.base_dir, bi_dir)
        n_files = 0
        paired = 0
        zh_only: Counter[str] = Counter()
        for rel, res in ctx.scans.items():
            self._abort_if_cancelled(ctx)
            by_int: dict[int, str] = {}
            depth = _brace_depths(res.vtex, [c.span.start for c in res.chunks])
            for c in res.chunks:
                cid = chunk_db_id(rel, c.span.start, c.span.end)
                zh = trans.get(cid)
                if zh is None:
                    continue
                if (
                    c.context == "para"
                    and c.env in _BI_ENVS
                    and depth[c.span.start] == 0
                ):
                    by_int[c.id] = _bi_text(c.content, zh)
                    paired += 1
                else:
                    by_int[c.id] = zh
                    zh_only[
                        c.env or ("{arg}" if depth[c.span.start] else c.context)
                    ] += 1
            if not by_int:
                continue
            (bi_dir / rel).write_text(
                reconstruct(res, by_int), encoding="utf-8", newline=""
            )
            n_files += 1
        self._log(
            ctx,
            f"bilingual: splice {n_files} files, paired {paired}, zh-only {dict(zh_only)}",
            force=True,
        )
        prepare_chinese(bi_dir, ctx.main_rel)
        main = bi_dir / ctx.main_rel
        tex = main.read_text(encoding="utf-8")
        out = _splice_before_document(tex, _BI_PREAMBLE, sentinel="TeXlateBiDefs")
        if out == tex:
            msg = f"{ctx.main_rel} 无 \\begin{{document}} 锚，无法注入对照样式宏"
            raise BilingualBuildError(msg)
        main.write_text(out, encoding="utf-8", newline="")
        res, v = compile_judge(
            self._engine(ctx),
            bi_dir,
            ctx.main_rel,
            timeout=self._compile_timeout,
            expect_cjk=ctx.expect_cjk,
            should_cancel=ctx.cancel_flag.is_set,
        )
        self._log(
            ctx,
            f"bilingual: verdict {v.status} cat={v.category} errs={v.n_errors}",
            force=True,
        )
        if not (res.has_pdf and res.pdf is not None):
            msg = f"对照版编译无 pdf（verdict={v.status} cat={v.category}）"
            raise BilingualBuildError(msg)
        dst = ctx.root / "bi.pdf"
        shutil.copyfile(res.pdf, dst)
        self._embed_tounicode(ctx, dst)
        self._register(ctx, "bi_pdf", "bi.pdf", force=True)


__all__ = ["BilingualBuildError", "_Bilingual"]
