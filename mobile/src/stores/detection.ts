import { create } from 'zustand'
import type { DetectionResult } from '@/types/detection'

interface DetectionState {
	imageUri: string | null
	result: DetectionResult | null
	setDetection: (payload: { imageUri: string; result: DetectionResult }) => void
	clearDetection: () => void
}

export const useDetectionStore = create<DetectionState>(
	(set) => ({
		imageUri: null,
		result: null,
		setDetection: ({ imageUri, result }) => set({ imageUri, result }),
		clearDetection: () => set({ imageUri: null, result: null }),
	}),
)
