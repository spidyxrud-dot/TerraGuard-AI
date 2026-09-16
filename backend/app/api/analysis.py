from fastapi import APIRouter

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/status")
def analysis_status() -> dict[str, str]:
    return {"status": "not_configured", "message": "Upload processing will be added in the GeoTIFF milestone."}
