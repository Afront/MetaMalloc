import os
from conans import ConanFile, tools, Meson

class MetaMallocConan(ConanFile):
	exports_sources = "src/*"
	generators='pkg_config'
	requires = 'tl-expected/20190710'
	settings = 'os', 'compiler', 'build_type', 'arch'

	def build(self):
		meson = Meson(self)
		# meson.configure(build_folder='.')
		meson.configure(
			build_folder=f'{self.source_folder}/build',
			source_folder=f'{self.source_folder}/src'
		)
		meson.build()
