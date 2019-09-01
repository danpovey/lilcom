#!/usr/bin/env python3


import numpy as np
import lilcom



def test_float():
    a = np.random.randn(100, 200).astype(np.float32)

    b = lilcom.compress(a)
    c = lilcom.decompress(b, dtype=np.float32)

    rel_error = (np.fabs(a - c)).sum() / (np.fabs(a)).sum()
    print("Relative error in float compression is: ", rel_error)

def test_int16():
    a = ((np.random.rand(100, 200) * 65535) - 32768).astype(np.int16)
    b = lilcom.compress(a)

    # decompressing as int16, float or double should give the same result except
    # it would be scaled by 1/32768
    for d in [np.int16, np.float32, np.float64]:
        c = lilcom.decompress(b, dtype=d)
        a2 = a.astype(np.float32) * (1.0/32768.0 if d != np.int16 else 1.0)
        c2 = c.astype(np.float32)
        rel_error = (np.fabs(a2 - c2)).sum() / (np.fabs(a2)).sum()
        print("Relative error in int16 compression (decompressing as {}) is {}".format(
                d, rel_error))

def test_double():
    a = np.random.randn(100, 200).astype(np.float64)

    b = lilcom.compress(a)
    c = lilcom.decompress(b, dtype=np.float64)

    rel_error = (np.fabs(a - c)).sum() / (np.fabs(a)).sum()
    print("Relative error in double compression, decompressing as double, is: ", rel_error)

    c = lilcom.decompress(b, dtype=np.float32)
    rel_error = (np.fabs(a - c)).sum() / (np.fabs(a)).sum()
    print("Relative error in double compression, decompressing as float, is: ", rel_error)



def main():
    test_float()
    test_int16()
    test_double()


if __name__ == "__main__":
    main()
