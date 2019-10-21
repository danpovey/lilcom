
CC ?= gcc


default: test

lilcom.o: lilcom.c

# Note: it's OK to use -O1, -O2, -O3, but it wouldn't really be correct to use
# -Ofast, because that will disable the checks for NaN's and inf's.

test: lilcom.c
	gcc -Wall -ftrapv -g -o test -DLILCOM_TEST=1 lilcom.c -o lilcom -lm
