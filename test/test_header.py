#!/usr/bin/env python3


import lilcom
import numpy as np

a = np.empty((1))
b = lilcom.compress(a)

assert b[0] == 76
assert b[1] == 0

print("Header begins with \"L0\"")

try:
    lilcom.decompress(bytes())
except ValueError as e:
    print(e.args[0])
    assert e.args[0] == "lilcom: Length of string was too short"
    pass

try:
    lilcom.decompress(bytes("X00", "utf-8"))
except ValueError as e:
    print(e.args[0])
    assert e.args[0] == "lilcom: Lilcom-compressed data must begin with L"
    pass

try:
    lilcom.decompress(bytes("L10", "utf-8"))
except ValueError as e:
    print(e.args[0])
    assert e.args[0] == "lilcom: Trying to decompress data from a future "\
                        "format version (use newer code)"
    pass
