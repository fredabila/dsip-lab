import { Text, View } from 'react-native'
import { CLASS_COLORS } from '@/constants/ppe'
import type { Detection } from '@/types/detection'

interface DetectionOverlayProps {
	detections: Detection[]
	frameWidth: number
	frameHeight: number
}

export function DetectionOverlay({
	detections,
	frameWidth,
	frameHeight,
}: DetectionOverlayProps) {
	return (
		<View className="absolute inset-0">
			{detections.map((detection, index) => {
				const { x, y, width, height } = detection.boundingBox
				const color = CLASS_COLORS[detection.className]

				const left = x * frameWidth
				const top = y * frameHeight
				const boxWidth = width * frameWidth
				const boxHeight = height * frameHeight

				return (
					<View
						// biome-ignore lint/suspicious/noArrayIndexKey: <explanation>
						key={index}
						className="absolute border-2"
						style={{
							left,
							top,
							width: boxWidth,
							height: boxHeight,
							borderColor: color,
						}}
					>
						<View
							style={{ backgroundColor: color }}
							className="px-1 py-0.5 self-start"
						>
							<Text className="text-white text-xs font-bold">
								{detection.className} {(detection.confidence * 100).toFixed(0)}%
							</Text>
						</View>
					</View>
				)
			})}
		</View>
	)
}
