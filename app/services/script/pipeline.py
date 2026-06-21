"""
Two-stage script generation pipeline.

Orchestrates:
    1. Template loading        → loads the JSON schema
    2. Planner (LLM call #1)   → generates structured Script Plan from template + topic
    3. Writer  (LLM call #2)   → generates final spoken script from Script Plan

Usage:
    from app.services.script import generate_script_with_template

    script = generate_script_with_template(
        topic="机会成本",
        template_id="finance_concept_v1",
        language="zh",
    )
"""

from loguru import logger

from .template import TemplateManager, get_template_manager
from .planner import generate_script_plan
from .writer import generate_final_script


def generate_script_with_template(
    topic: str,
    template_id: str = "",
    language: str = "",
    template_manager: TemplateManager | None = None,
) -> str:
    """Generate a video script using the two-stage Planner + Writer pipeline.

    Args:
        topic: The video subject / topic.
        template_id: ID of the template to use (e.g. "finance_concept_v1").
                     If empty, the first available template is used.
        language: Output language for the script.
        template_manager: Optional TemplateManager instance (uses singleton if None).

    Returns:
        The final video script as a plain string, or an error string starting
        with "Error: " on failure.
    """
    if template_manager is None:
        template_manager = get_template_manager()

    # 1. Load template
    if not template_id:
        template_id = template_manager.get_default_template_id()
        if not template_id:
            return "Error: no templates available. Please create a template first."

    try:
        template = template_manager.load_template(template_id)
    except FileNotFoundError as e:
        return f"Error: {str(e)}"

    logger.info(
        f"starting two-stage script generation: "
        f"topic={topic}, template={template_id}"
    )

    # 2. Generate Script Plan (Planner)
    plan = generate_script_plan(
        topic=topic,
        template=template,
        language=language,
        template_manager=template_manager,
    )

    if isinstance(plan, str) and plan.startswith("Error:"):
        return plan

    # 3. Generate final script (Writer)
    script = generate_final_script(
        script_plan=plan,
        template=template,
        language=language,
    )

    return script
