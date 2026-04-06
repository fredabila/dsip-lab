// src/app/_layout.tsx
import { Stack } from 'expo-router'
import { HeroUINativeProvider } from 'heroui-native'
import { GestureHandlerRootView } from 'react-native-gesture-handler'
import {
	SafeAreaListener,
	SafeAreaProvider,
} from 'react-native-safe-area-context'
import { Uniwind } from 'uniwind'
import '../index.css'

export default function RootLayout() {
	return (
		<GestureHandlerRootView style={{ flex: 1 }}>
			<SafeAreaProvider>
				<SafeAreaListener
					onChange={({ insets }) => Uniwind.updateInsets(insets)}
				>
					<HeroUINativeProvider>
						<Stack screenOptions={{ headerShown: false }} />
					</HeroUINativeProvider>
				</SafeAreaListener>
			</SafeAreaProvider>
		</GestureHandlerRootView>
	)
}
