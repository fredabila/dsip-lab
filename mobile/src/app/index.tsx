// src/app/index.tsx
import { useMutation } from '@tanstack/react-query'
import * as Burnt from 'burnt'
import * as ImagePicker from 'expo-image-picker'
import { useFocusEffect, useRouter } from 'expo-router'
import { Spinner } from 'heroui-native/spinner'
import { useCallback, useEffect, useRef, useState } from 'react'
import { StyleSheet, Text, View } from 'react-native'
import {
	Camera,
	useCameraDevice,
	useCameraPermission,
} from 'react-native-vision-camera'
import { CameraControls } from '@/components/camera/camera-controls'
import { CameraPermissionDialog } from '@/components/camera/camera-permission-dialog'
import { ImageAnalysisOverlay } from '@/components/camera/image-analysis-overlay'
import { ScanStats } from '@/components/camera/scan-stats'
import { DetectionOverlay } from '@/components/detection-overlay'
import { IdleOverlay } from '@/components/idle-overlay'
import { VIOLATION_CLASSES } from '@/constants/ppe'
import { detectPPE } from '@/services/detection'
import { useDetectionStore } from '@/stores/detection'
import type { DetectionResult } from '@/types/detection'

const FRAME_INTERVAL_MS = 500
const CONSECUTIVE_FAILURE_THRESHOLD = 3
const IDLE_TIMEOUT_MS = 15_000

export default function CameraScreen() {
	const router = useRouter()
	const setDetection = useDetectionStore((state) => state.setDetection)
	const cameraRef = useRef<Camera>(null)
	const inFlight = useRef<boolean>(false)
	const lastActivityRef = useRef<number>(Date.now())
	const { hasPermission, requestPermission } = useCameraPermission()

	const [permissionDenied, setPermissionDenied] = useState(false)
	const [isAnalyzing, setIsAnalyzing] = useState(false)
	const [isTorchOn, setIsTorchOn] = useState(false)
	const [cameraPosition, setCameraPosition] = useState<'back' | 'front'>('back')
	const [isPaused, setIsPaused] = useState(false)
	const [isRequestingPermission, setIsRequestingPermission] = useState(false)
	const [frameSize, setFrameSize] = useState({ width: 0, height: 0 })
	const [detectionResult, setDetectionResult] =
		useState<DetectionResult | null>(null)

	const hasFirstResult = detectionResult !== null
	const consecutiveFailures = useRef(0)

	const device = useCameraDevice(cameraPosition)

	const { mutateAsync: runDetection } = useMutation({
		mutationFn: detectPPE,
		onSuccess: (data) => {
			consecutiveFailures.current = 0
			if (data.detections.length > 0) {
				lastActivityRef.current = Date.now()
				setIsPaused(false)
			}
			setDetectionResult(data)
		},
		onError: () => {
			consecutiveFailures.current += 1
			if (consecutiveFailures.current >= CONSECUTIVE_FAILURE_THRESHOLD) {
				Burnt.toast({
					title: 'Detection unavailable',
					message: 'Check your connection and try again',
					preset: 'error',
					haptic: 'error',
					from: 'bottom',
				})
				consecutiveFailures.current = 0
			}
		},
	})

	const { mutateAsync: runImageDetection } = useMutation({
		mutationFn: detectPPE,
		onError: () => {
			Burnt.toast({
				title: 'Detection unavailable',
				message: 'Check your connection and try again',
				preset: 'error',
				haptic: 'error',
				from: 'bottom',
			})
		},
	})

	useEffect(() => {
		const interval = setInterval(() => {
			if (Date.now() - lastActivityRef.current > IDLE_TIMEOUT_MS) {
				setIsPaused(true)
			}
		}, 1000)
		return () => clearInterval(interval)
	}, [])

	useEffect(() => {
		if (!hasPermission || !device || isPaused) return

		let cancelled = false
		let timeoutId: ReturnType<typeof setTimeout> | null = null

		async function loop() {
			if (cancelled || !cameraRef.current) return

			if (inFlight.current) {
				timeoutId = setTimeout(loop, FRAME_INTERVAL_MS)
				return
			}

			try {
				inFlight.current = true
				const photo = await cameraRef.current.takeSnapshot({ quality: 60 })
				if (!cancelled) {
					await runDetection(photo.path)
				}
			} catch {
			} finally {
				inFlight.current = false
			}

			if (!cancelled) {
				timeoutId = setTimeout(loop, FRAME_INTERVAL_MS)
			}
		}

		loop()

		return () => {
			cancelled = true
			if (timeoutId) {
				clearTimeout(timeoutId)
			}
		}
	}, [hasPermission, device, runDetection, isPaused])

	useFocusEffect(
		useCallback(() => {
			lastActivityRef.current = Date.now()
			setIsPaused(false)
		}, []),
	)

	const handleRequestPermission = async () => {
		if (isRequestingPermission) return

		setIsRequestingPermission(true)
		try {
			const granted = await requestPermission()
			if (!granted) {
				setPermissionDenied(true)
				Burnt.toast({
					title: 'Camera access needed',
					message: 'Enable camera permission to start detection',
					preset: 'error',
					haptic: 'error',
					from: 'bottom',
				})
			}
		} catch {
			Burnt.toast({
				title: 'Permission request failed',
				message: 'Try again from your device settings',
				preset: 'error',
				haptic: 'error',
				from: 'bottom',
			})
		} finally {
			setIsRequestingPermission(false)
		}
	}

	if (!hasPermission) {
		return (
			<CameraPermissionDialog
				isRequestingPermission={isRequestingPermission}
				isDenied={permissionDenied}
				onRequestPermission={handleRequestPermission}
			/>
		)
	}

	if (!device) {
		return (
			<View className="flex-1 items-center justify-center bg-background">
				<Text className="text-foreground text-base">
					No camera device found
				</Text>
			</View>
		)
	}

	const violationCount =
		detectionResult?.detections.filter((d) =>
			VIOLATION_CLASSES.includes(d.className),
		).length ?? 0

	const compliantCount =
		detectionResult?.detections.filter(
			(d) => !VIOLATION_CLASSES.includes(d.className),
		).length ?? 0

	const handleImagePicker = async () => {
		try {
			const permission = await ImagePicker.requestMediaLibraryPermissionsAsync()
			if (!permission.granted) {
				Burnt.toast({
					title: 'Permission denied',
					message: 'Allow photo library access to upload images',
					preset: 'error',
					haptic: 'error',
					from: 'bottom',
				})
				return
			}

			const result = await ImagePicker.launchImageLibraryAsync({
				mediaTypes: ['images'],
				allowsEditing: false,
				quality: 1,
			})

			if (result.canceled || !result.assets?.[0]) return

			const asset = result.assets[0]
			setIsAnalyzing(true)

			const data = await runImageDetection(asset.uri).catch(() => null)
			if (!data) return

			setDetection({ imageUri: asset.uri, result: data })
			router.push('/result')
		} catch {
			Burnt.toast({
				title: 'Could not open image picker',
				message: 'Please try again',
				preset: 'error',
				haptic: 'error',
				from: 'bottom',
			})
		} finally {
			setIsAnalyzing(false)
		}
	}

	return (
		<View
			className="flex-1 bg-black"
			onTouchStart={() => (lastActivityRef.current = Date.now())}
		>
			<Camera
				ref={cameraRef}
				torch={isTorchOn ? 'on' : 'off'}
				style={StyleSheet.absoluteFill}
				device={device}
				isActive={!isPaused}
				photo
				onLayout={(e) => {
					const { width, height } = e.nativeEvent.layout
					setFrameSize({ width, height })
				}}
			/>

			{detectionResult && frameSize.width > 0 && (
				<DetectionOverlay
					detections={detectionResult.detections}
					frameWidth={frameSize.width}
					frameHeight={frameSize.height}
				/>
			)}

			<View className="absolute top-0 left-0 right-0 items-center pt-safe-offset-4">
				<ScanStats
					compliantCount={compliantCount}
					violationCount={violationCount}
				/>
			</View>

			{!hasFirstResult && (
				<View
					className="absolute inset-0 items-center justify-center bg-overlay"
					accessibilityRole="progressbar"
					accessibilityLabel="Connecting to detection service"
				>
					<Spinner size="lg" />
					<Text
						className="text-foreground mt-3 text-sm"
						accessibilityLiveRegion="polite"
					>
						Connecting...
					</Text>
				</View>
			)}

			<View className="absolute bottom-0 left-0 right-0 items-center pb-safe-offset-6">
				<CameraControls
					isTorchOn={isTorchOn}
					onTorchToggle={() => setIsTorchOn((v) => !v)}
					onFlipCamera={() =>
						setCameraPosition((v) => (v === 'back' ? 'front' : 'back'))
					}
					onGalleryPress={handleImagePicker}
				/>
			</View>

			{isPaused && (
				<IdleOverlay
					onResume={() => {
						lastActivityRef.current = Date.now()
						setIsPaused(false)
					}}
				/>
			)}

			{isAnalyzing && <ImageAnalysisOverlay />}
		</View>
	)
}
