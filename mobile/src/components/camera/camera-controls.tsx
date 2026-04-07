// src/components/camera/camera-controls.tsx

import { Pressable, View } from 'react-native'
import FlipIcon from '@/components/icons/lucide/flip-horizontal-2'
import GalleryIcon from '@/components/icons/lucide/image'
import FlashlightIcon from '@/components/icons/lucide/zap'

interface CameraControlsProps {
	isTorchOn: boolean
	onTorchToggle: () => void
	onFlipCamera: () => void
	onGalleryPress: () => void
}

export function CameraControls({
	isTorchOn,
	onTorchToggle,
	onFlipCamera,
	onGalleryPress,
}: CameraControlsProps) {
	return (
		<View className="flex-row items-center bg-black/80 border border-white/10 rounded-full px-2 py-2">
			<Pressable
				onPress={onTorchToggle}
				className="w-11 h-11 rounded-full items-center justify-center"
				accessibilityRole="button"
				accessibilityLabel="Toggle flashlight"
			>
				<FlashlightIcon
					width={20}
					height={20}
					color={isTorchOn ? '#facc15' : 'rgba(255,255,255,0.9)'}
				/>
			</Pressable>

			<View className="w-px h-5 bg-white/10 mx-1" />

			<Pressable
				onPress={onFlipCamera}
				className="w-11 h-11 rounded-full items-center justify-center"
				accessibilityRole="button"
				accessibilityLabel="Flip camera"
			>
				<FlipIcon width={20} height={20} color="rgba(255,255,255,0.9)" />
			</Pressable>

			<View className="w-px h-5 bg-white/10 mx-1" />

			<Pressable
				onPress={onGalleryPress}
				className="w-11 h-11 rounded-full items-center justify-center"
				accessibilityRole="button"
				accessibilityLabel="Open gallery"
			>
				<GalleryIcon width={20} height={20} color="rgba(255,255,255,0.9)" />
			</Pressable>
		</View>
	)
}
