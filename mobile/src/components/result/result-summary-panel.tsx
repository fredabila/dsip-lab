// src/components/result/result-summary-panel.tsx
import { useRouter } from 'expo-router'
import { Separator } from 'heroui-native'
import { Button } from 'heroui-native/button'
import { Text, View } from 'react-native'
import CheckCircleIcon from '@/components/icons/lucide/circle-check'
import XCircleIcon from '@/components/icons/lucide/circle-x'
import { useDetectionStore } from '@/stores/detection'

interface ResultSummaryPanelProps {
	compliantCount: number
	violationCount: number
	inferenceTimeMs: number
}

export function ResultSummaryPanel({
	compliantCount,
	violationCount,
	inferenceTimeMs,
}: ResultSummaryPanelProps) {
	const router = useRouter()
	const clearDetection = useDetectionStore((s) => s.clearDetection)

	const formatInferenceMs = (value: number) => {
		if (!Number.isFinite(value) || value < 0) return '--'
		return String(Math.round(value))
	}

	return (
		<View className="px-4 pt-4 pb-safe-offset-4 bg-overlay gap-4">
			{/* Stats */}
			<View className="gap-2">
				<Text className="text-foreground/70 text-sm uppercase tracking-wide">
					Result Summary
				</Text>

				<View className="bg-background/45 rounded-2xl px-4 py-5 gap-2">
					<View className="flex-row justify-between py-2">
						<View className="flex-row items-center gap-1.5">
							<CheckCircleIcon width={15} height={15} color="#4ade80" />
							<Text className="text-green-400 font-semibold">
								{Math.max(0, compliantCount)} Compliant
							</Text>
						</View>
						<View className="flex-row items-center gap-1.5">
							<Text className="text-red-400 font-semibold">
								{Math.max(0, violationCount)}{' '}
								{violationCount === 1 ? 'Violation' : 'Violations'}
							</Text>
							<XCircleIcon width={15} height={15} color="#f87171" />
						</View>
					</View>

					<Separator className="bg-foreground/10" />

					<View className="flex-row justify-between">
						<Text className="text-foreground/80 text-xs uppercase">Inference time</Text>
						<Text className="text-foreground/80 text-xs">
							{formatInferenceMs(inferenceTimeMs)} MS
						</Text>
					</View>
				</View>
			</View>

			{/* Action */}
			<Button
				variant="outline"
				size="lg"
				className="w-full border-white/20 bg-white/10 py-3 rounded-3xl"
				onPress={() => {
					clearDetection()
					router.back()
				}}
			>
				<Button.Label className="text-white/80 text-xl font-semibold uppercase tracking-wide">
					Scan another
				</Button.Label>
			</Button>
		</View>
	)
}
