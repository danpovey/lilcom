from distutils.core import setup, Extension

extension_mod = Extension("lilcom", ["lilcommodule.c", "lilcom.c"])

setup(name = "lilcom", ext_modules=[extension_mod])

