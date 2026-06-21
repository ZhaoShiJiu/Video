"""
Template manager for script generation templates.

Loads JSON template schemas from the templates/ directory and provides
rendering utilities for prompt construction.
"""

import json
import os
from typing import Dict, List, Optional

from loguru import logger

# Root directory of the project (two levels up from this file)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
_TEMPLATES_DIR = os.path.join(_PROJECT_ROOT, "templates")


class TemplateManager:
    """Manages script templates stored as JSON files in the templates/ directory."""

    def __init__(self, templates_dir: str = _TEMPLATES_DIR):
        self._templates_dir = templates_dir
        self._cache: Dict[str, dict] = {}

    @property
    def templates_dir(self) -> str:
        return self._templates_dir

    def list_templates(self) -> List[dict]:
        """List all available templates with their metadata."""
        templates = []
        if not os.path.isdir(self._templates_dir):
            logger.warning(f"templates directory not found: {self._templates_dir}")
            return templates

        for filename in sorted(os.listdir(self._templates_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self._templates_dir, filename)
            try:
                template = self._load_template_file(filepath)
                templates.append(
                    {
                        "id": template.get("id", ""),
                        "name": template.get("name", ""),
                        "description": template.get("description", ""),
                        "duration": template.get("duration", 60),
                    }
                )
            except Exception as e:
                logger.warning(f"failed to load template {filename}: {e}")
        return templates

    def load_template(self, template_id: str) -> dict:
        """Load a template by its ID. Cached after first load."""
        if template_id in self._cache:
            return self._cache[template_id]

        filepath = os.path.join(self._templates_dir, f"{template_id}.json")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"template '{template_id}' not found at {filepath}"
            )

        template = self._load_template_file(filepath)
        self._cache[template_id] = template
        return template

    def render_template_for_prompt(self, template: dict) -> str:
        """Render a template schema into human-readable text for the LLM prompt.

        Produces output like:

            脚本结构：

            第一部分 Hook（0-5秒）
            目标：
            制造认知冲突

            要求：
            - 不要直接解释概念
            - 使用反常识观点

            ...

        The rendered text separates structure (which the LLM must follow)
        from the runtime context (topic, language) that the PromptBuilder
        handles separately.
        """
        if not template:
            return ""

        steps = template.get("steps", [])
        if not steps:
            return ""

        lines = ["脚本结构：", ""]
        for i, step in enumerate(steps, 1):
            key = step.get("key", f"step_{i}")
            duration = step.get("duration", "")
            goal = step.get("goal", "")
            rules = step.get("rules", [])

            # Section header
            label_parts = [f"第{i}部分 {key.capitalize()}"]
            if duration:
                label_parts.append(f"（{duration}）")
            lines.append("".join(label_parts))

            if goal:
                lines.append(f"目标：")
                lines.append(goal)

            if rules:
                lines.append("")
                lines.append("要求：")
                for rule in rules:
                    lines.append(f"- {rule}")

            lines.append("")
            lines.append("")

        return "\n".join(lines)

    def get_step_keys(self, template: dict) -> List[str]:
        """Return the ordered list of step keys from a template."""
        steps = template.get("steps", [])
        return [step.get("key", "") for step in steps]

    def get_default_template_id(self) -> str:
        """Return the first available template ID, or empty string if none."""
        templates = self.list_templates()
        if not templates:
            return ""
        return templates[0]["id"]

    @staticmethod
    def _load_template_file(filepath: str) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            template = json.load(f)
        _validate_template(template, filepath)
        return template

    def clear_cache(self):
        """Clear the template cache (useful for hot-reload during development)."""
        self._cache.clear()


def _validate_template(template: dict, filepath: str):
    """Validate that a template has the required fields."""
    if not template.get("id"):
        raise ValueError(f"template missing 'id' field: {filepath}")
    if not template.get("steps"):
        raise ValueError(f"template missing 'steps' array: {filepath}")
    for i, step in enumerate(template["steps"]):
        if not step.get("key"):
            raise ValueError(
                f"template '{template['id']}' step {i} missing 'key' field"
            )


# Singleton instance
_template_manager: Optional[TemplateManager] = None


def get_template_manager() -> TemplateManager:
    """Get the singleton TemplateManager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager
