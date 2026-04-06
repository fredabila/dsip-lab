import { QueryClient } from '@tanstack/react-query'
import * as Burnt from 'burnt'

export const queryClient = new QueryClient({
	defaultOptions: {
		queries: {
			retry: 2,
			staleTime: 1000 * 60,
		},
		mutations: {
			onError: (error: Error) => {
				Burnt.toast({
					title: 'Something went wrong',
					message: error?.message ?? 'Please try again',
					preset: 'error',
					haptic: 'error',
					duration: 3,
					from: 'bottom',
				})
			},
		},
	},
})
