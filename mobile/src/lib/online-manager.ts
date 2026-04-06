import { onlineManager } from '@tanstack/react-query'
import * as Network from 'expo-network'

export function setupOnlineManager() {
	onlineManager.setEventListener((setOnline) => {
		let initialised = false

		const subscription = Network.addNetworkStateListener((state) => {
			initialised = true
			setOnline(!!state.isConnected)
		})

		// Handle initial state before first listener event fires
		Network.getNetworkStateAsync()
			.then((state) => {
				if (!initialised) setOnline(!!state.isConnected)
			})
			.catch(() => {
				// getNetworkStateAsync can reject on some SDK versions
			})

		return subscription.remove
	})
}
