"""
Planner — first LLM call in the two-stage pipeline.

Generates a structured Script Plan (JSON) from:
- a video topic
- a fixed template schema (injected into the prompt)
"""

import json
import logging
import re

from loguru import logger

from app.services.llm import generate_json_response
from .prompt_builder import build_planner_prompt, PLANNER_SYSTEM_PROMPT
from .template import TemplateManager, get_template_manager

_MAX_RETRIES = 5


def _strip_code_fence(text: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def generate_script_plan(
    topic: str,
    template: dict,
    language: str = "",
    template_manager: TemplateManager | None = None,
) -> dict | str:
    """Generate a structured Script Plan from a topic and template.

    This is the first LLM call in the two-stage pipeline. It injects the
    template schema as a system prompt and returns a JSON dict mapping each
    template step key to its core idea.

    Returns:
        A dict like {"hook": "...", "scenario": "...", ...} on success,
        or an error string starting with "Error: " on failure.
    """
    if template_manager is None:
        template_manager = get_template_manager()

    user_prompt = build_planner_prompt(
        topic=topic,
        template=template,
        language=language,
        template_manager=template_manager,
    )

    logger.info(
        f"generating script plan: topic={topic}, template={template.get('id', 'unknown')}"
    )

    step_keys = template_manager.get_step_keys(template)
    plan = {}
    for attempt in range(_MAX_RETRIES):
        try:
            raw = generate_json_response(user_prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
            if raw.startswith("Error:"):
                logger.error(f"planner LLM error: {raw}")
                if attempt < _MAX_RETRIES - 1:
                    continue
                return raw

            # Parse and validate
            parsed = json.loads(_strip_code_fence(raw))

            # Validate that all expected keys are present
            missing = [k for k in step_keys if k not in parsed]
            if missing:
                logger.warning(
                    f"planner output missing keys: {missing}, retrying..."
                )
                if attempt < _MAX_RETRIES - 1:
                    continue

            # Keep only the expected keys + allow extras
            for key in step_keys:
                if key in parsed:
                    plan[key] = str(parsed[key])

            if plan:
                break

        except json.JSONDecodeError as e:
            logger.warning(
                f"planner returned invalid JSON (attempt {attempt + 1}): {e}"
            )
            if attempt < _MAX_RETRIES - 1:
                continue
            return f"Error: planner returned invalid JSON after {_MAX_RETRIES} attempts"

        except Exception as e:
            logger.error(f"planner error: {e}")
            if attempt < _MAX_RETRIES - 1:
                continue
            return f"Error: {str(e)}"

    if not plan:
        return f"Error: failed to generate script plan after {_MAX_RETRIES} attempts"

    logger.success(f"script plan generated: {json.dumps(plan, ensure_ascii=False)}")
    return plan
