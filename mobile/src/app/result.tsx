import { useRouter } from 'expo-router'
import { Button } from 'heroui-native/button'
import { useState } from 'react'
import { Image, Text, View } from 'react-native'
import { DetectionOverlay } from '@/components/detection-overlay'
import { VIOLATION_CLASSES } from '@/constants/ppe'
import { useDetectionStore } from '@/stores/detection'

export default function ResultScreen() {
	const router = useRouter()
	const imageUri = useDetectionStore((s) => s.imageUri)
	const result = useDetectionStore((s) => s.result)
	const clearDetection = useDetectionStore((s) => s.clearDetection)

	const [frameSize, setFrameSize] = useState({ width: 0, height: 0 })

	const detections = result?.detections ?? []

	const violationCount = detections.filter((d) =>
		VIOLATION_CLASSES.includes(d.className),
	).length

	const compliantCount = detections.length - violationCount

	if (!imageUri || !result) {
		return (
			<View className="flex-1 items-center justify-center bg-black">
				<Text className="text-white">No detection result available</Text>
				<Button
					variant="primary"
					onPress={() => router.back()}
					className="mt-4"
				>
					Go Back
				</Button>
			</View>
		)
	}

	return (
		<View className="flex-1 bg-black">
			<View
				className="flex-1"
				onLayout={(e) => {
					const { width, height } = e.nativeEvent.layout
					setFrameSize({ width, height })
				}}
			>
				{imageUri ? (
					<>
						<Image
							source={{ uri: imageUri }}
							className="w-full h-full"
							resizeMode="contain"
						/>
						{frameSize.width > 0 && frameSize.height > 0 && (
							<DetectionOverlay
								detections={detections}
								frameWidth={frameSize.width}
								frameHeight={frameSize.height}
							/>
						)}
					</>
				) : (
					<View className="flex-1 items-center justify-center">
						<Text className="text-white">No image selected</Text>
					</View>
				)}
			</View>

			<View className="px-4 py-4 bg-black/85 gap-3">
				<View className="flex-row justify-between">
					<Text className="text-green-400 font-semibold">
						✓ {compliantCount} Compliant
					</Text>
					<Text className="text-red-400 font-semibold">
						✕ {violationCount} Violation{violationCount !== 1 ? 's' : ''}
					</Text>
				</View>

				<Text className="text-white/80 text-xs">
					Inference: {result.inferenceTimeMs.toFixed(0)} ms
				</Text>

				<Button
					variant="primary"
					onPress={() => {
						clearDetection()
						router.back()
					}}
				>
					Scan another
				</Button>
			</View>
		</View>
	)
}
