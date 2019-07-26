
CC ?= gcc


default: test

lilcom.o: lilcom.c

test: lilcom.c lilcom.o
	gcc -o test lilcom.o





