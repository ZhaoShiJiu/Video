"""
AI Image Tagging Management Page

Streamlit page for:
- Tag statistics overview
- Batch tagging configuration & progress
- Tag distribution charts
- Quick tag-based search

All service calls are direct Python imports — no HTTP to self.
"""
import os
import sys
import threading
import time

import streamlit as st

# Ensure project root is in path
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

from app.config import config
from app.models import const
from app.services import llm
from app.services import state as sm
from app.services import tagging
from app.utils import utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_material_dir() -> str:
    """Get the configured material directory."""
    _md = config.app.get("material_directory", "").strip()
    if _md:
        return _md
    return utils.storage_dir("local_videos", create=True)


def _get_candidates() -> dict:
    """Get candidate lists for filter dropdowns."""
    try:
        return llm.get_tag_candidates()
    except Exception:
        return {"characters": [], "emotions": [], "events": []}


def _start_batch_tagging(task_id: str, base_dir: str, force: bool, max_concurrent: int):
    """Run batch tagging in a background thread so the UI stays responsive."""
    try:
        tagging.batch_tag_images(
            base_dir=base_dir,
            force=force,
            max_concurrent=max_concurrent,
            task_id=task_id,
        )
    except Exception as e:
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            progress=100,
            errors=[{"file": "", "error": str(e)}],
        )


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render():
    st.title("🤖 AI 图片打标管理")

    if not config.tagging.get("enabled", True):
        st.warning("AI 打标功能未启用。请在 config.toml 中设置 [tagging] enabled = true")
        return

    base_dir = _get_material_dir()

    # ---- Overview ----
    st.subheader("📊 概览")

    try:
        stats = tagging.get_tag_statistics(base_dir)
    except Exception as e:
        stats = {}
        st.warning(f"无法加载统计数据: {e}")

    if stats:
        cols = st.columns(4)
        cols[0].metric("总图片", stats.get("total_images", 0))
        cols[1].metric("已打标", stats.get("tagged_count", 0))
        cols[2].metric("未打标", stats.get("untagged_count", 0))
        cols[3].metric("平均标签数", stats.get("avg_tags_per_image", 0))
    else:
        st.info("暂无统计数据。请先进行打标操作。")

    st.divider()

    # ---- Tagging Config ----
    st.subheader("🔧 打标设置")

    c1, c2 = st.columns(2)
    with c1:
        st.text(f"模型: {config.tagging.get('vision_model', 'qwen3-vl-flash')}")
    with c2:
        max_concurrent = st.selectbox(
            "并发数",
            options=[1, 2, 3, 4, 5],
            index=min(config.tagging.get("max_concurrent", 3) - 1, 4),
            help="并发调用 API 的线程数，过高可能触发限流",
        )

    force = st.radio(
        "打标策略",
        options=[False, True],
        index=0,
        format_func=lambda x: "🔄 强制重新打标所有图片" if x else "✅ 仅处理未打标/已变更的图片（推荐）",
    )

    # ---- Start tagging ----
    if st.button("🚀 开始打标", type="primary", use_container_width=True):
        task_id = utils.get_uuid()
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_PROCESSING,
            progress=0,
            total=0,
            tagged=0,
            skipped=0,
            failed=0,
            current_file="",
            errors=[],
        )
        t = threading.Thread(
            target=_start_batch_tagging,
            args=(task_id, base_dir, force, max_concurrent),
            daemon=True,
        )
        t.start()
        st.session_state["tagging_task_id"] = task_id
        st.success(f"打标任务已启动: {task_id}")
        st.rerun()

    # ---- Progress ----
    task_id = st.session_state.get("tagging_task_id", "")
    if task_id:
        st.divider()
        st.subheader("📈 进度")

        task = sm.state.get_task(task_id)
        if task:
            state_val = task.get("state")
            progress = task.get("progress", 0)
            total = task.get("total", 0)
            tagged = task.get("tagged", 0)
            skipped = task.get("skipped", 0)
            failed = task.get("failed", 0)
            current_file = task.get("current_file", "")
            errors = task.get("errors", [])

            st.progress(min(1.0, max(0.0, progress / 100.0)))

            state_map = {
                const.TASK_STATE_PROCESSING: "🔄 运行中",
                const.TASK_STATE_COMPLETE: "✅ 已完成",
                const.TASK_STATE_FAILED: "❌ 失败",
            }
            st.markdown(
                f"**{state_map.get(state_val, str(state_val))}** | "
                f"进度: {progress}% ({tagged + skipped + failed}/{total})\n\n"
                f"当前: {current_file or '—'}\n\n"
                f"成功: {tagged}　跳过: {skipped}　失败: {failed}"
            )

            if errors:
                with st.expander(f"错误详情 ({len(errors)})"):
                    for err in errors:
                        st.text(f"❌ {err.get('file', '?')}: {err.get('error', '?')}")

            done = state_val in (const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED)
            if done:
                if st.button("清除任务状态"):
                    st.session_state.pop("tagging_task_id", None)
                    st.rerun()
            else:
                time.sleep(1.5)
                st.rerun()
        else:
            st.warning(f"未找到任务: {task_id}")
            st.session_state.pop("tagging_task_id", None)

    st.divider()

    # ---- Distributions ----
    if stats:
        st.subheader("🏷 标签分布")

        t1, t2, t3, t4 = st.tabs(["👤 角色", "😨 情绪", "🎬 事件", "🎨 颜色"])

        with t1:
            cd = stats.get("character_distribution", {})
            st.bar_chart(cd, height=300) if cd else st.info("暂无数据")

        with t2:
            ed = stats.get("emotion_distribution", {})
            st.bar_chart(ed, height=300) if ed else st.info("暂无数据")

        with t3:
            evd = stats.get("event_distribution", {})
            st.bar_chart(evd, height=300) if evd else st.info("暂无数据")

        with t4:
            cod = stats.get("color_distribution", {})
            st.bar_chart(cod, height=300) if cod else st.info("暂无数据")

    st.divider()

    # ---- Search ----
    st.subheader("🔍 快速搜索")

    candidates = _get_candidates()
    cc, ce, cev = st.columns(3)
    with cc:
        sel_chars = st.multiselect("角色", options=candidates.get("characters", []), key="s_char")
    with ce:
        sel_emos = st.multiselect("情绪", options=candidates.get("emotions", []), key="s_emo")
    with cev:
        sel_evts = st.multiselect("事件", options=candidates.get("events", []), key="s_evt")

    keyword = st.text_input("描述关键词", placeholder="输入关键词模糊搜索...")

    match_mode = st.selectbox(
        "匹配模式", options=["any", "all"], index=0,
        format_func=lambda x: "任意匹配 (OR)" if x == "any" else "全部匹配 (AND)",
    )

    if st.button("🔍 搜索", type="secondary"):
        try:
            results = tagging.search_materials_by_tags(
                base_dir=base_dir,
                characters=sel_chars if sel_chars else None,
                emotions=sel_emos if sel_emos else None,
                events=sel_evts if sel_evts else None,
                keyword=keyword.strip() if keyword else None,
                match_mode=match_mode,
            )
        except Exception as e:
            st.error(f"搜索失败: {e}")
            results = []

        st.caption(f"匹配结果: {len(results)} 张")

        if results:
            cols_per_row = 4
            for i in range(0, len(results), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col in enumerate(row_cols):
                    idx = i + j
                    if idx >= len(results):
                        break
                    r = results[idx]
                    with col:
                        with st.container(border=True):
                            st.text(r["file_path"])
                            st.caption(f"👤 {', '.join(r.get('characters', [])) or '—'}")
                            st.caption(f"😨 {', '.join(r.get('emotions', [])) or '—'}")
                            st.caption(f"🎬 {', '.join(r.get('events', [])) or '—'}")
                            sc = r.get("match_score", 0)
                            st.caption(f"匹配度: {'★' * sc}{'☆' * max(0, 5 - sc)}")
                            with st.expander("详情"):
                                st.write(f"**描述:** {r.get('description', '—')}")
                                st.write(f"**颜色:** {', '.join(r.get('colors', []))}")
                                st.json(r.get("match_detail", {}))
        else:
            st.info("未找到匹配的素材")


if __name__ == "__main__":
    render()
