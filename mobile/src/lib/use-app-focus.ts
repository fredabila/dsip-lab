import { focusManager } from '@tanstack/react-query'
import { useEffect } from 'react'
import type { AppStateStatus } from 'react-native'
import { AppState } from 'react-native'

function onAppStateChange(status: AppStateStatus) {
	focusManager.setFocused(status === 'active')
}

export function useAppFocus() {
	useEffect(() => {
		const subscription = AppState.addEventListener('change', onAppStateChange)
		return () => subscription.remove()
	}, [])
}
