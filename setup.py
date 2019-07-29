from distutils.core import setup, Extension

extension_mod = Extension("lilcomlib", ["lilcommodule.c"])

setup(name = "lilcomlib", ext_modules=[extension_mod])
