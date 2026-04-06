// src/lib/utils.ts
import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs))
}

export async function photoToBase64(path: string): Promise<string> {
	const blob = await fetch(`file://${path}`).then((r) => r.blob())
	return new Promise((resolve, reject) => {
		const reader = new FileReader()
		reader.onloadend = () => {
			const result = reader.result as string
			resolve(result.split(',')[1] ?? '')
		}
		reader.onerror = reject
		reader.readAsDataURL(blob)
	})
}
