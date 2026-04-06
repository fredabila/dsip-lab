from fastapi import APIRouter, File, HTTPException, UploadFile

from schemas.detection import DetectionResponse
from services.inference import ppe_service

router = APIRouter(prefix="/api", tags=["detection"])


@router.post("/detect", response_model=DetectionResponse)
async def detect(frame: UploadFile = File(...)) -> DetectionResponse:
	if not frame.content_type or not frame.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="File must be an image")

	image_bytes = await frame.read()

	try:
		return ppe_service.run(image_bytes)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
