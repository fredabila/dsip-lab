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
import { QueryClientProvider } from '@tanstack/react-query'
import { setupOnlineManager } from '@/lib/online-manager'
import { queryClient } from '@/lib/query-client'
import { useAppFocus } from '@/lib/use-app-focus'

setupOnlineManager()

export default function RootLayout() {
	useAppFocus()
	return (
		<GestureHandlerRootView style={{ flex: 1 }}>
			<SafeAreaProvider>
				<SafeAreaListener
					onChange={({ insets }) => Uniwind.updateInsets(insets)}
				>
					<HeroUINativeProvider config={{ devInfo: { stylingPrinciples: false } }}>
						<QueryClientProvider client={queryClient}>
							<Stack screenOptions={{ headerShown: false }} />
						</QueryClientProvider>
					</HeroUINativeProvider>
				</SafeAreaListener>
			</SafeAreaProvider>
		</GestureHandlerRootView>
	)
}
