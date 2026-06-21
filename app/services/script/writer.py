"""
Writer — second LLM call in the two-stage pipeline.

Generates the final spoken video script from the structured Script Plan.
The template is NOT injected here — it's already embodied in the plan.
"""

import logging
import re

from loguru import logger

from app.services.llm import generate_text_response
from .prompt_builder import build_writer_prompt, WRITER_SYSTEM_PROMPT

_MAX_RETRIES = 5


def _format_script(raw: str) -> str:
    """Clean up the Writer output into a publishable script."""
    # Remove markdown formatting
    raw = raw.replace("*", "")
    raw = raw.replace("#", "")
    raw = re.sub(r"\[.*?\]", "", raw)
    raw = re.sub(r"\(.*?\)", "", raw)

    # Split into paragraphs
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return "\n\n".join(paragraphs)


def generate_final_script(
    script_plan: dict,
    template: dict | None = None,
    language: str = "",
) -> str:
    """Generate the final spoken script from a Script Plan.

    This is the second LLM call in the two-stage pipeline. The template is
    not re-injected — it has already been encoded in the plan structure.

    Returns:
        The final script as a plain string, or an error string starting
        with "Error: " on failure.
    """
    user_prompt = build_writer_prompt(
        script_plan=script_plan,
        template=template,
        language=language,
    )

    logger.info("generating final script from plan")

    final_script = ""
    for attempt in range(_MAX_RETRIES):
        try:
            raw = generate_text_response(
                user_prompt,
                system_prompt=WRITER_SYSTEM_PROMPT,
            )
            if raw.startswith("Error:"):
                logger.error(f"writer LLM error: {raw}")
                if attempt < _MAX_RETRIES - 1:
                    continue
                return raw

            final_script = _format_script(raw)
            if final_script:
                break

        except Exception as e:
            logger.error(f"writer error (attempt {attempt + 1}): {e}")
            if attempt < _MAX_RETRIES - 1:
                continue
            return f"Error: {str(e)}"

    if not final_script:
        return f"Error: failed to generate final script after {_MAX_RETRIES} attempts"

    logger.success(f"final script generated: {final_script[:80]}...")
    return final_script.strip()
