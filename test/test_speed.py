#!/usr/bin/env python3


import numpy as np
import lilcom
import time


def test_rtf():
    for dtype in [np.float32, np.float64]:

        test_duration = 0.2

        for tick_power in [-8,-6,-4]:
            flops = 0
            a = np.random.randn(300,300).astype(dtype)

            start = time.process_time()
            while time.process_time() - start < test_duration:
                flops += a.size
                a = np.random.randn(*a.shape).astype(dtype)
            print("Flops/sec for randn with dtype={} is {} ".format(
                dtype, flops / (time.process_time() - start)))


            start = time.process_time()
            while time.process_time() - start < test_duration:
                flops += a.size
                b = lilcom.compress(a, tick_power=tick_power)
            print("Flops/sec for compression with dtype={} and tick_power={} is {} ".format(
                dtype, tick_power, flops / (time.process_time() - start)))

            start = time.process_time()
            while time.process_time() - start < test_duration:
                flops += a.size
                a2 = lilcom.decompress(b)
            print("Flops/sec for decompression with tick_power={} is {}".format(
                tick_power, flops / (time.process_time() - start)))





def main():
    test_rtf()


if __name__ == "__main__":
    main()
