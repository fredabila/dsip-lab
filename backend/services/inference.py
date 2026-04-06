import time
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import io

from schemas.detection import Detection, BoundingBox, DetectionResponse

MODEL_PATH = Path(__file__).parent.parent / "models" / "helmax_ppe.pt"

VIOLATION_CLASSES = {"no-helmet", "no-vest", "no-boots"}
COMPLIANT_CLASSES = {"helmet", "vest", "boots"}

CONFIDENCE_THRESHOLD = 0.35


class PPEInferenceService:
	def __init__(self):
		self._model: YOLO | None = None

	def load(self):
		if self._model is None:
			self._model = YOLO(str(MODEL_PATH))

	def run(self, image_bytes: bytes) -> DetectionResponse:
		if self._model is None:
			raise RuntimeError("Model not loaded. Call load() first.")

		image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

		start = time.perf_counter()
		results = self._model.predict(
			source=image,
			conf=CONFIDENCE_THRESHOLD,
			verbose=False,
		)
		inference_time_ms = (time.perf_counter() - start) * 1000

		detections: list[Detection] = []

		for result in results:
			boxes = result.boxes
			if boxes is None:
				continue
			
			for box in boxes:
				class_index = int(box.cls[0])
				class_name = result.names[class_index]
				confidence = float(box.conf[0])

				x1, y1, x2, y2 = box.xyxyn[0].tolist()

				detections.append(
					Detection(
						class_index=class_index,
						class_name=class_name,
						confidence=confidence,
						bounding_box=BoundingBox(
							x=x1,
							y=y1,
							width=x2 - x1,
							height=y2 - y1,
						),
					)
				)

		violation_count = sum(1 for d in detections if d.class_name in VIOLATION_CLASSES)
		compliant_count = sum(1 for d in detections if d.class_name in COMPLIANT_CLASSES)

		return DetectionResponse(
			detections=detections,
			inference_time_ms=inference_time_ms,
			violation_count=violation_count,
			compliant_count=compliant_count,
		)


ppe_service = PPEInferenceService()
