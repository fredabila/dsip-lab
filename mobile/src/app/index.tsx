// src/app/index.tsx
import { useMutation } from '@tanstack/react-query'
import * as Burnt from 'burnt'
import { Spinner } from 'heroui-native/spinner'
import { useEffect, useRef, useState } from 'react'
import { StyleSheet, Text, View } from 'react-native'
import {
	Camera,
	useCameraDevice,
	useCameraPermission,
} from 'react-native-vision-camera'
import { DetectionOverlay } from '@/components/detection-overlay'
import { IdleOverlay } from '@/components/idle-overlay'
import { VIOLATION_CLASSES } from '@/constants/ppe'
import { detectPPE } from '@/services/detection'
import type { DetectionResult } from '@/types/detection'

const FRAME_INTERVAL_MS = 500
const CONSECUTIVE_FAILURE_THRESHOLD = 3
const IDLE_TIMEOUT_MS = 15_000

export default function CameraScreen() {
	const { hasPermission, requestPermission } = useCameraPermission()
	const device = useCameraDevice('back')
	const cameraRef = useRef<Camera>(null)
	const inFlight = useRef<boolean>(false)
	const lastActivityRef = useRef<number>(Date.now())

	const [isPaused, setIsPaused] = useState(false)
	const [frameSize, setFrameSize] = useState({ width: 0, height: 0 })
	const [detectionResult, setDetectionResult] =
		useState<DetectionResult | null>(null)

	const hasFirstResult = detectionResult !== null
	const consecutiveFailures = useRef(0)

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

		async function loop() {
			if (cancelled || !cameraRef.current) return

			if (inFlight.current) {
				setTimeout(loop, FRAME_INTERVAL_MS)
				return
			}

			try {
				inFlight.current = true
				const photo = await cameraRef.current.takeSnapshot({ quality: 60 })
				if (!cancelled) {
					await runDetection(photo.path)
				}
			} catch (e) {
				console.log('loop error:', e)
			} finally {
				inFlight.current = false
			}

			if (!cancelled) {
				setTimeout(loop, FRAME_INTERVAL_MS)
			}
		}

		loop()

		return () => {
			cancelled = true
		}
	}, [hasPermission, device, runDetection, isPaused])

	if (!hasPermission) {
		return (
			<View className="flex-1 items-center justify-center bg-black">
				<Text className="text-white text-base" onPress={requestPermission}>
					Camera permission required — tap to grant
				</Text>
			</View>
		)
	}

	if (!device) {
		return (
			<View className="flex-1 items-center justify-center bg-black">
				<Text className="text-white text-base">No camera device found</Text>
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

	return (
		<View
			className="flex-1 bg-black"
			onTouchStart={() => (lastActivityRef.current = Date.now())}
		>
			<Camera
				ref={cameraRef}
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

			<View className="absolute top-0 left-0 right-0 flex-row justify-between items-center px-4 pt-safe pb-2 bg-black/50">
				<Text className="text-white font-bold text-base">Helmax</Text>
				<View className="flex-row gap-3">
					<Text className="text-green-400 font-semibold text-sm">
						✓ {compliantCount} Compliant
					</Text>
					<Text className="text-red-400 font-semibold text-sm">
						✕ {violationCount} Violation{violationCount !== 1 ? 's' : ''}
					</Text>
				</View>
			</View>

			{!hasFirstResult && (
				<View className="absolute inset-0 items-center justify-center bg-black/70">
					<Spinner size="lg" />
					<Text className="text-white mt-3 text-sm">Connecting...</Text>
				</View>
			)}

			{isPaused && (
				<IdleOverlay
					onResume={() => {
						lastActivityRef.current = Date.now()
						setIsPaused(false)
					}}
				/>
			)}
		</View>
	)
}
