#!/usr/bin/env python3


import lilcom
import numpy as np

for shape in [ (40,50), (3,4,5), (1,5,7), (8,1,10), (100,2,57) ]:
    a = np.random.randn(*shape)
    for power in [ -15, -8, -6 ]:
        b = lilcom.compress(a, power)
        a2 = lilcom.decompress(b)
        print("len(b) = ", len(b), ", bytes per number = ", (len(b) / a.size))

        assert lilcom.get_shape(b) == a2.shape

        diff = (a2 - a)
        mx = diff.max()
        mn = diff.min()
        limit = (2 ** (power-1)) + 5.0e-05  # add a small margin to account for
                                            # floating point roundoff.
        print("max,min diff = {}, {}, expected magnitude was {}".format(mx, mn, limit))
        assert mx <= limit and -mn <= limit
