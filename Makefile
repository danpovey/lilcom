
CC ?= gcc


default: test

lilcom.o: lilcom.c

test: lilcom.c
	gcc -o test lilcom.c



