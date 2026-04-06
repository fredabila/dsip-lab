// src/constants/ppe.ts
export const PPE_CLASSES = [
	'boots',
	'gloves',
	'goggles',
	'helmet',
	'no-boots',
	'no-gloves',
	'no-goggles',
	'no-helmet',
	'no-vest',
	'vest',
] as const

export type PPEClass = (typeof PPE_CLASSES)[number]

export const VIOLATION_CLASSES: PPEClass[] = [
	'no-helmet',
	'no-vest',
	'no-boots',
]

export const COMPLIANT_CLASSES: PPEClass[] = ['helmet', 'vest', 'boots']

export const CLASS_COLORS: Record<PPEClass, string> = {
	helmet: '#00C853',
	vest: '#00C853',
	boots: '#00C853',
	'no-helmet': '#FF1744',
	'no-vest': '#FF1744',
	'no-boots': '#FF1744',
	gloves: '#2196F3',
	'no-gloves': '#FF9800',
	goggles: '#2196F3',
	'no-goggles': '#FF9800',
}
