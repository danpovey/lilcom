#!/usr/bin/env python3


import numpy as np
import lilcom
import time


def test_rtf():
    for dtype in [np.int16, np.float32, np.float64]:
        # view the following as 100 channels where each channel
        # is one second's worth of 16kHz-sampled data.
        audio_time = 1000.0;  # seconds
        a = np.random.randn(int(audio_time), 16000)
        if dtype == np.int16:
            a *= 32768;
        a = a.astype(dtype)
        for bits_per_sample in [4,8]:
            for lpc_order in [0,1,2,4,8]:
                for axis in [0, 1]:
                    start = time.process_time()
                    b = lilcom.compress(a, axis=axis,
                                        bits_per_sample=bits_per_sample,
                                        lpc_order=lpc_order)
                    mid = time.process_time()
                    c = lilcom.decompress(b, dtype=dtype)
                    end = time.process_time()

                    # f is a factor that we'll multiply the times by.  The
                    # factor of 100.0 is to make the output percentages.
                    f = 100.0 / audio_time

                    print("RTF for dtype={}, bits-per-sample={}, lpc_order={}, axis={}, "
                          "compress/decompress/total RTF is: {:.3f}%,{:.3f}%,{:.3f}%".format(
                            dtype, bits_per_sample, lpc_order, axis,
                            (mid - start) * f, (end - mid) * f, (end - start) * f))



def main():
    test_rtf()


if __name__ == "__main__":
    main()
