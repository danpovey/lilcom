
CC ?= gcc


default: test

lilcom.o: lilcom.c

test: lilcom.c
	gcc -g -o test -DLILCOM_TEST=1 lilcom.c -lm
