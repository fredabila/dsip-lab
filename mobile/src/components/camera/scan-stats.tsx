// src/components/camera/scan-stats.tsx

import { Text, View } from 'react-native'
import AlertCircleIcon from '@/components/icons/lucide/circle-alert'
import CheckCircleIcon from '@/components/icons/lucide/circle-check'

interface ScanStatsProps {
	compliantCount: number
	violationCount: number
}

export function ScanStats({ compliantCount, violationCount }: ScanStatsProps) {
	return (
		<View className="items-center">
			<View className="flex-row items-center gap-4 bg-black/80 border border-white/10 rounded-full px-5 py-2.5">
				<View className="flex-row items-center gap-1.5">
					<CheckCircleIcon width={15} height={15} color="#4ade80" />
					<Text className="text-green-400 text-sm font-medium">
						{compliantCount} Compliant
					</Text>
				</View>
				<View className="w-px h-3.5 bg-white/15" />
				<View className="flex-row items-center gap-1.5">
					<AlertCircleIcon width={15} height={15} color="#f87171" />
					<Text className="text-red-400 text-sm font-medium">
						{violationCount} Violation{violationCount !== 1 ? 's' : ''}
					</Text>
				</View>
			</View>
		</View>
	)
}
