// src/services/detection.ts

import { fetch } from 'expo/fetch'
import { File } from 'expo-file-system'
import { DETECT_ENDPOINT } from '@/constants/api'
import type { PPEClass } from '@/constants/ppe'
import type { DetectionResult } from '@/types/detection'

interface RawDetection {
	class_index: number
	class_name: string
	confidence: number
	bounding_box: {
		x: number
		y: number
		width: number
		height: number
	}
}

interface RawDetectionResponse {
	detections: RawDetection[]
	inference_time_ms: number
	violation_count: number
	compliant_count: number
}

export async function detectPPE(photoPath: string): Promise<DetectionResult> {
	const normalizedPath = photoPath.startsWith('file://')
		? photoPath
		: `file://${photoPath}`

	const file = new File(normalizedPath)
	const formData = new FormData()
	formData.append('frame', file)

	const response = await fetch(DETECT_ENDPOINT, {
		method: 'POST',
		body: formData,
		headers: {
			'ngrok-skip-browser-warning': 'true',
		},
	})

	if (!response.ok) {
		throw new Error(`Detection failed: ${response.status}`)
	}

	const raw: RawDetectionResponse = await response.json()

	return {
		detections: raw.detections.map((d) => ({
			classIndex: d.class_index,
			className: d.class_name as PPEClass,
			confidence: d.confidence,
			boundingBox: {
				x: d.bounding_box.x,
				y: d.bounding_box.y,
				width: d.bounding_box.width,
				height: d.bounding_box.height,
			},
		})),
		inferenceTimeMs: raw.inference_time_ms,
	}
}
