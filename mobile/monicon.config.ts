import type { MoniconConfig } from '@monicon/core'
import { clean, reactNative } from '@monicon/core/plugins'

export default {
	icons: ['mdi:home', 'feather:activity'],
	collections: ['lucide',],
	plugins: [
		clean({ patterns: ['src/components/icons'] }),
		reactNative({ outputPath: 'src/components/icons' }),
	],
} satisfies MoniconConfig
