from pydantic import BaseModel


class BoundingBox(BaseModel):
	x: float
	y: float
	width: float
	height: float


class Detection(BaseModel):
	class_index: int
	class_name: str
	confidence: float
	bounding_box: BoundingBox


class DetectionResponse(BaseModel):
	detections: list[Detection]
	inference_time_ms: float
	violation_count: int
	compliant_count: int
