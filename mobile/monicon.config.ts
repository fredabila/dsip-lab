import type { MoniconConfig } from '@monicon/core'
import { clean, reactNative } from '@monicon/core/plugins'

export default {
	icons: ['mdi:home', 'feather:activity'],
	collections: ['lucide',],
	plugins: [
		clean({ patterns: ['components/icons'] }),
		reactNative({ outputPath: 'components/icons' }),
	],
} satisfies MoniconConfig
