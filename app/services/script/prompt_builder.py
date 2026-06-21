"""
Prompt builders for the two-stage script generation pipeline.

- build_planner_prompt():  injects the template schema and topic into a planner prompt
- build_writer_prompt():   converts the Script Plan into a writer prompt
"""

import json
from typing import Dict

from .template import TemplateManager, get_template_manager

# ---------------------------------------------------------------------------
# Planner prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """你是一名顶级财经短视频策划。

你的任务不是写完整文案，而是根据给定的脚本结构，设计每个部分的核心内容。

你必须：
1. 严格按照模板的步骤输出。
2. 每个步骤只写核心思想，不写完整台词。
3. 输出严格 JSON。
4. 每个步骤的内容控制在 1-2 句话，精炼有力。"""


def build_planner_prompt(
    topic: str,
    template: dict,
    language: str = "",
    template_manager: TemplateManager | None = None,
) -> str:
    """Build the full user prompt for the Planner LLM.

    Combines the topic and the rendered template schema into a single
    prompt that instructs the LLM to produce a structured Script Plan.
    """
    if template_manager is None:
        template_manager = get_template_manager()

    rendered = template_manager.render_template_for_prompt(template)
    step_keys = template_manager.get_step_keys(template)

    prompt = f"""主题：
{topic}

{rendered}
请根据以上脚本结构，为主题「{topic}」生成 Script Plan。

输出一个 JSON 对象，包含以下字段：
{json.dumps(step_keys, ensure_ascii=False)}

只返回 JSON 对象，不要包含任何其他文字。"""
    if language:
        prompt += f"\n\n使用语言：{language}"

    return prompt


# ---------------------------------------------------------------------------
# Writer prompts
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = """你是一名拥有百万粉丝的财经博主。
擅长用通俗、口语化、有冲突感的方式解释经济学知识。"""


def build_writer_prompt(
    script_plan: dict,
    template: dict | None = None,
    language: str = "",
) -> str:
    """Build the user prompt for the Writer LLM.

    Converts the structured Script Plan (output of the Planner) into a
    prompt that instructs the Writer to produce the final spoken script.

    The template is NOT injected here — it's already embodied in the plan.
    Including template metadata (like duration) is optional context.
    """
    duration_hint = ""
    if template and template.get("duration"):
        duration_hint = f"这是一个大约 {template['duration']} 秒的短视频。"

    # Pretty-print the plan for the LLM
    plan_text = _render_plan_for_writer(script_plan)

    prompt = f"""根据下面的脚本规划，写一个短视频口播稿。

{duration_hint}
要求：
- 像朋友聊天
- 多用短句
- 有停顿感
- 不要像教材
- 大约250~350字
- 直接输出口播稿，不要加任何说明文字

Script Plan:

{plan_text}"""
    if language:
        prompt += f"\n\n使用语言：{language}"

    return prompt


def _render_plan_for_writer(plan: dict) -> str:
    """Pretty-print a Script Plan dict for the Writer prompt."""
    if not plan:
        return "{}"

    lines = []
    for key, value in plan.items():
        if isinstance(value, str):
            lines.append(f"{key}: {value}")
        elif isinstance(value, dict):
            # Nested plan with sub-fields
            sub_lines = [f"{key}:"]
            for sub_key, sub_val in value.items():
                sub_lines.append(f"  {sub_key}: {sub_val}")
            lines.append("\n".join(sub_lines))
        else:
            lines.append(f"{key}: {value}")
        lines.append("")

    return "\n".join(lines)
