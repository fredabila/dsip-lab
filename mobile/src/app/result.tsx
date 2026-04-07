import { Image } from 'expo-image'
import { useRouter } from 'expo-router'
import { useState } from 'react'
import { View } from 'react-native'
import { useResolveClassNames } from 'uniwind'
import { DetectionOverlay } from '@/components/detection-overlay'
import { QueryError } from '@/components/query-error'
import { ResultSummaryPanel } from '@/components/result/result-summary-panel'
import { VIOLATION_CLASSES } from '@/constants/ppe'
import { useDetectionStore } from '@/stores/detection'

export default function ResultScreen() {
	const router = useRouter()
	const imageUri = useDetectionStore((s) => s.imageUri)
	const result = useDetectionStore((s) => s.result)

	const imageStyle = useResolveClassNames('w-full h-full')

	const [hasImageLoadError, setHasImageLoadError] = useState(false)
	const [frameSize, setFrameSize] = useState({ width: 0, height: 0 })

	const detections = result?.detections ?? []

	const violationCount = detections.filter((d) =>
		VIOLATION_CLASSES.includes(d.className),
	).length

	const compliantCount = detections.length - violationCount

	if (!imageUri || !result) {
		return (
			<QueryError
				className="bg-background"
				message="No detection result is available."
				onRetry={() => router.back()}
			/>
		)
	}

	return (
		<View className="flex-1 bg-background">
			<View
				className="flex-1"
				onLayout={(e) => {
					const { width, height } = e.nativeEvent.layout
					if (width !== frameSize.width || height !== frameSize.height) {
						setFrameSize({ width, height })
					}
				}}
			>
				{imageUri && !hasImageLoadError ? (
					<>
						<Image
							source={{ uri: imageUri }}
							style={imageStyle}
							contentFit="contain"
							onError={() => setHasImageLoadError(true)}
							accessibilityLabel="Detection result image"
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
					<QueryError
						className="bg-background"
						message={
							hasImageLoadError
								? 'We could not load that image. Please try scanning another photo.'
								: 'No image selected.'
						}
						onRetry={() => router.back()}
					/>
				)}
			</View>

			<ResultSummaryPanel
				compliantCount={compliantCount}
				violationCount={violationCount}
				inferenceTimeMs={result.inferenceTimeMs}
			/>
		</View>
	)
}
