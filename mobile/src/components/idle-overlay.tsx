// src/components/idle-overlay.tsx
import { Text, TouchableOpacity } from 'react-native'

interface IdleOverlayProps {
	onResume: () => void
}

export function IdleOverlay({ onResume }: IdleOverlayProps) {
	return (
		<TouchableOpacity
			className="absolute inset-0 items-center justify-center bg-black/80"
			onPress={onResume}
			activeOpacity={1}
		>
			<Text className="text-white text-lg font-bold">Paused</Text>
			<Text className="text-white/60 text-sm mt-1">Tap to resume scanning</Text>
		</TouchableOpacity>
	)
}
