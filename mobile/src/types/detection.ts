// src/types/detection.ts
import type { PPEClass } from '@/constants/ppe'

export interface BoundingBox {
	x: number
	y: number
	width: number
	height: number
}

export interface Detection {
	classIndex: number
	className: PPEClass
	confidence: number
	boundingBox: BoundingBox
}

export interface DetectionResult {
	detections: Detection[]
	inferenceTimeMs: number
}
