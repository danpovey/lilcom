from distutils.core import setup, Extension

extension_mod = Extension("spam", ["spammodule.c"])

setup(name = "spam", ext_modules=[extension_mod])
