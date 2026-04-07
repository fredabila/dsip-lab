import { BlurView } from 'expo-blur'
import { Spinner } from 'heroui-native/spinner'
import { StyleSheet, Text } from 'react-native'

export function ImageAnalysisOverlay() {
  return (
    <BlurView
      intensity={80}
      tint="dark"
      style={StyleSheet.absoluteFill}  
      className="items-center justify-center"
    >
      <Spinner size="lg" />
      <Text className="text-white text-sm font-medium mt-3">
        Analyzing image…
      </Text>
    </BlurView>
  )
}