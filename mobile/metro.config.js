// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config')
const { withUniwindConfig } = require('uniwind/metro')
const { withMonicon } = require('@monicon/metro')

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname)

const moniconConfig = withMonicon(config)

module.exports = withUniwindConfig(moniconConfig, {
  cssEntryFile: './src/index.css',
  dtsFile: './src/uniwind-types.d.ts'
})