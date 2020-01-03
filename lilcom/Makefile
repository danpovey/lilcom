
CC ?= gcc


default: lilcom fixed_math

lilcom.o: lilcom.cc

fixed_math.o: fixed_math.c


# Note: it's OK to use -O1, -O2, -O3, but it wouldn't really be correct to use
# -Ofast, because that will disable the checks for NaN's and inf's.
# I was getting mysterious "illegal instruction" errors with -ftrapv that
# i had trouble
lilcom: lilcom.c lilcom.h bit_packer.c bit_packer.h encoder.c encoder.h lilcom_common.h
	g++ -O0 -Wall -g -DLILCOM_TEST=1 lilcom.c -o lilcom -lm # -ftrapv


fixed_math: fixed_math.c
	g++ -O0 -Wall -g -o fixed_math -DFIXED_MATH_TEST=1 fixed_math.c -o fixed_math -lm # -ftrapv


prediction_math: prediction_math.c fixed_math.c prediction_math.h fixed_math.h
	g++ -O0 -Wall -g  -DPREDICTION_MATH_TEST=1 fixed_math.c prediction_math.c -o prediction_math -lm # -ftrapv

int_math_utils_test: int_math_utils_test.cc int_math_utils.h
	g++ -O0 -Wall -g  int_math_utils_test.cc -o int_math_utils_test -lm # -ftrapv

int_scalar_test: int_scalar_test.cc int_scalar.h int_math_utils.h
	g++ -O0 -Wall -g  int_scalar_test.cc -o int_scalar_test -lm # -ftrapv

int_vec_test: int_vec_test.cc int_scalar.h int_vec.h int_math_utils.h
	g++ -O0 -Wall -g  int_vec_test.cc -o int_vec_test -lm # -ftrapv

lpc_math_test: lpc_math_test.cc lpc_math.cc lpc_math.h int_scalar.h int_vec.h int_math_utils.h
	g++ -O0 -Wall -g  lpc_math.cc lpc_math_test.cc -o lpc_math_test -lm # -ftrapv

bit_stream_test: bit_stream_test.cc bit_stream.h
	g++ -O0 -Wall -g  bit_stream_test.cc -o bit_stream_test -lm # -ftrapv

int_stream_test: int_stream_test.cc int_stream.h bit_stream.h
	g++ -O0 -Wall -g  int_stream_test.cc -o int_stream_test -lm # -ftrapv
