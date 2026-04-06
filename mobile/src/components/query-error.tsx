import { Pressable, Text, View } from 'react-native'
import { cn } from '@/lib/utils'

interface QueryErrorProps {
	message?: string
	onRetry?: () => void
	className?: string
}

export function QueryError({ message, onRetry, className }: QueryErrorProps) {
	return (
		<View
			className={cn('flex-1 items-center justify-center gap-3 p-6', className)}
		>
			<Text className="text-destructive text-base font-medium text-center">
				{message ?? 'Something went wrong'}
			</Text>
			{onRetry && (
				<Pressable
					onPress={onRetry}
					className="px-4 py-2 rounded-lg bg-destructive/10 active:opacity-70"
				>
					<Text className="text-destructive text-sm font-medium">
						Try again
					</Text>
				</Pressable>
			)}
		</View>
	)
}
