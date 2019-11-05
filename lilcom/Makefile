
CC ?= gcc


default: test

lilcom.o: lilcom.cc

# Note: it's OK to use -O1, -O2, -O3, but it wouldn't really be correct to use
# -Ofast, because that will disable the checks for NaN's and inf's.
# I was getting mysterious "illegal instruction" errors with -ftrapv that
# i had trouble
test: lilcom.cc
	g++ -O0 -Wall -g -o test -DLILCOM_TEST=1 lilcom.cc -o lilcom -lm # -ftrapv
