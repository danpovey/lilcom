#!/usr/bin/env python3


import numpy as np
import lilcom


def test_float():
    for bits_per_sample in [4,6,8]:
        for axis in [-1, 1, 0, -2]:
            for use_out in [False, True]:
                a = np.random.randn(100+bits_per_sample+axis,
                                    200+bits_per_sample+axis).astype(np.float32)
                out_shape = lilcom.get_compressed_shape(a.shape, axis, bits_per_sample)

                b = lilcom.compress(a, axis=axis, bits_per_sample=bits_per_sample,
                                    out=(np.empty(out_shape, dtype=np.int8) if use_out else None))
                c = lilcom.decompress(b, dtype=(None if use_out else np.float32),
                                      out=(np.empty(a.shape, dtype=np.float32) if use_out else None))

            rel_error = (np.fabs(a - c)).sum() / (np.fabs(a)).sum()
            print("Relative error in float compression (axis={}, bits-per-sample={}) is {}".format(
                    axis, bits_per_sample, rel_error))

def test_int16():
    for bits_per_sample in [4,5,8]:
        for axis in [-1, 1, 0, -2]:
            a = ((np.random.rand(100 + bits_per_sample + axis,
                                 200 + 10*bits_per_sample + axis) * 65535) - 32768).astype(np.int16)
            for use_out in [False, True]:
                out_shape = lilcom.get_compressed_shape(a.shape, axis, bits_per_sample)

                b = lilcom.compress(a, axis=axis, bits_per_sample=bits_per_sample,
                                    out=(np.empty(out_shape, dtype=np.int8) if use_out else None))
                # decompressing as int16, float or double should give the same result except
                # it would be scaled by 1/32768
                for d in [np.int16, np.float32, np.float64]:
                    c = lilcom.decompress(b,
                                          dtype=(None if use_out else d),
                                          out=(np.empty(a.shape, dtype=d) if use_out else None))

                    a2 = a.astype(np.float32) * (1.0/32768.0 if d != np.int16 else 1.0)
                    c2 = c.astype(np.float32)
                    rel_error = (np.fabs(a2 - c2)).sum() / (np.fabs(a2)).sum()
                    print("Relative error in int16 compression (decompressing as {}, axis={}, num-bits={}, use_out={}) is {}".format(
                            d, axis, bits_per_sample, use_out, rel_error))


def test_int16_lpc_order():
    a = ((np.random.rand(100, 200) * 65535) - 32768).astype(np.int16)

    for lpc in range(0, 15):
        b = lilcom.compress(a, axis=-1, lpc_order=lpc)

        c = lilcom.decompress(b, dtype=np.int16)

        a2 = a.astype(np.float32)
        c2 = c.astype(np.float32)

        rel_error = (np.fabs(a2 - c2)).sum() / (np.fabs(a2)).sum()
        print("Relative error in int16 with lpc order={} is {}".format(
                lpc, rel_error))

def test_double():
    a = np.random.randn(100, 200).astype(np.float64)

    b = lilcom.compress(a, axis=-1)
    c = lilcom.decompress(b, dtype=np.float64)

    rel_error = (np.fabs(a - c)).sum() / (np.fabs(a)).sum()
    print("Relative error in double compression, decompressing as double, is: ", rel_error)

    c = lilcom.decompress(b, dtype=np.float32)
    rel_error = (np.fabs(a - c)).sum() / (np.fabs(a)).sum()
    print("Relative error in double compression, decompressing as float, is: ", rel_error)



def main():
    test_int16()
    test_float()
    test_int16_lpc_order()
    test_double()


if __name__ == "__main__":
    main()
