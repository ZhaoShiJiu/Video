from fastapi import Request

from app.controllers.v1.base import new_router
from app.models.schema import (
    VideoScriptRequest,
    VideoScriptResponse,
    VideoSocialMetadataRequest,
    VideoSocialMetadataResponse,
    VideoTermsRequest,
    VideoTermsResponse,
)
from app.services import llm
from app.services.script import generate_script_with_template, get_template_manager
from app.utils import utils

# authentication dependency
# router = new_router(dependencies=[Depends(base.verify_token)])
router = new_router()


@router.post(
    "/scripts",
    response_model=VideoScriptResponse,
    summary="Create a script for the video",
)
def generate_video_script(request: Request, body: VideoScriptRequest):
    template_id = getattr(body, "template_id", "") or ""

    if template_id:
        # Two-stage Planner + Writer pipeline with template
        video_script = generate_script_with_template(
            topic=body.video_subject,
            template_id=template_id,
            language=body.video_language,
        )
    else:
        # Classic single-call pipeline
        video_script = llm.generate_script(
            video_subject=body.video_subject,
            language=body.video_language,
            paragraph_number=body.paragraph_number,
            video_script_prompt=body.video_script_prompt,
            custom_system_prompt=body.custom_system_prompt,
        )

    response = {"video_script": video_script}
    return utils.get_response(200, response)


@router.get(
    "/templates",
    summary="List available script templates",
)
def list_templates(request: Request):
    """Return all available script templates with their metadata."""
    tm = get_template_manager()
    templates = tm.list_templates()
    return utils.get_response(200, templates)


@router.post(
    "/terms",
    response_model=VideoTermsResponse,
    summary="Generate video terms based on the video script",
)
def generate_video_terms(request: Request, body: VideoTermsRequest):
    video_terms = llm.generate_terms(
        video_subject=body.video_subject,
        video_script=body.video_script,
        amount=body.amount,
    )
    response = {"video_terms": video_terms}
    return utils.get_response(200, response)


@router.post(
    "/social-metadata",
    response_model=VideoSocialMetadataResponse,
    summary="Generate social publishing metadata",
)
def generate_video_social_metadata(
    request: Request, body: VideoSocialMetadataRequest
):
    metadata = llm.generate_social_metadata(
        video_subject=body.video_subject,
        video_script=body.video_script,
        language=body.language,
        platform=body.platform,
    )
    return utils.get_response(200, metadata)
