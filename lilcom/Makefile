
# Note: this Makefile is just for testing.
# you would do
# cd ..; python3 ./setup.py build
# to actually build the python module.

CC ?= gcc


default: test 



# Note: it's OK to use -O1, -O2, -O3, but it wouldn't really be correct to use
# -Ofast, because that will disable the checks for NaN's and inf's.
# I was getting mysterious "illegal instruction" errors with -ftrapv that
# i had trouble

test: bit_stream_test int_stream_test
	for t in $^; do echo "Testing $$t"; ./$$t || exit 1; echo "*** Tested $$t; success ***"; sleep 1; done


clean: 
	-rm bit_stream_test int_stream_test


bit_stream_test: bit_stream_test.cc bit_stream.h
	g++ -O0 -Wall -g  bit_stream_test.cc -o bit_stream_test -lm # -ftrapv

int_stream_test: int_stream_test.cc int_stream.h bit_stream.h
	g++ -O0 -Wall -g  int_stream_test.cc -o int_stream_test -lm # -ftrapv
