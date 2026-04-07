// src/components/camera/camera-permission-dialog.tsx
import { Button } from 'heroui-native/button'
import { Dialog } from 'heroui-native/dialog'
import { Linking, View } from 'react-native'

interface CameraPermissionDialogProps {
	isRequestingPermission: boolean
	isDenied: boolean
	onRequestPermission: () => void
}

export function CameraPermissionDialog({
	isRequestingPermission,
	isDenied,
	onRequestPermission,
}: CameraPermissionDialogProps) {
	const handlePress = isDenied
		? () => Linking.openSettings()
		: onRequestPermission

	return (
		<View className="flex-1 bg-black">
			<Dialog isOpen onOpenChange={() => {}}>
				<Dialog.Portal>
					<Dialog.Overlay className="bg-black/60" />
					<Dialog.Content
						className="mx-5 rounded-xl bg-zinc-900 ring-1 ring-white/10 gap-0 overflow-hidden p-0"
						isSwipeable={false}
					>
						<View className="gap-1.5 p-4">
							<Dialog.Title className="text-white text-base font-medium">
								Camera access required
							</Dialog.Title>
							<Dialog.Description className="text-sm text-white/50">
								{isDenied
									? 'Enable camera access in Settings to continue.'
									: 'Allow camera access to scan PPE compliance.'}
							</Dialog.Description>
						</View>

						<View className="flex-row justify-end bg-white/5 border-t border-white/8 px-4 py-3">
							<Button
								variant="outline"
								size="sm"
								onPress={handlePress}
								isDisabled={isRequestingPermission}
								className="border-white/20 bg-white/10 p-3 rounded-3xl"
							>
								<Button.Label className="text-white/80 text-sm">
									{isDenied
										? 'Open Settings'
										: isRequestingPermission
											? 'Requesting…'
											: 'Grant access'}
								</Button.Label>
							</Button>
						</View>
					</Dialog.Content>
				</Dialog.Portal>
			</Dialog>
		</View>
	)
}