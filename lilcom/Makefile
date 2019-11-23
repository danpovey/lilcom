
CC ?= gcc


default: lilcom fixed_math

lilcom.o: lilcom.cc

fixed_math.o: fixed_math.c

# Note: it's OK to use -O1, -O2, -O3, but it wouldn't really be correct to use
# -Ofast, because that will disable the checks for NaN's and inf's.
# I was getting mysterious "illegal instruction" errors with -ftrapv that
# i had trouble
lilcom: lilcom.c
	gcc -O0 -Wall -g -DLILCOM_TEST=1 lilcom.c -o lilcom -lm # -ftrapv


fixed_math: fixed_math.c
	gcc -O0 -Wall -g -o fixed_math -DFIXED_MATH_TEST=1 fixed_math.c -o fixed_math -lm # -ftrapv
