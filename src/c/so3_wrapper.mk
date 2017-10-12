libso3f.so : so3_functions.o
	gcc -Wall -Werror -shared -o libso3f.so so3_functions.o -L/mn/stornext/u3/adriaand/so3/lib/c/ -lso3 -L/mn/stornext/u3/adriaand/ssht/lib/c -L/mn/stornext/u3/adriaand/ssht/lib/c -lssht -L/mn/stornext/u3/adriaand/fftw/lib -lfftw3 -lm

so3_functions.o : so3_functions.c
	gcc -g -Wall -Wextra -fpic -c so3_functions.c -I/mn/stornext/u3/adriaand/so3/include/c/ -I/mn/stornext/u3/adriaand/ssht/include/c/ -I/mn/stornext/u3/adriaand/fftw/include -L/mn/stornext/u3/adriaand/so3/lib/c/ -lso3 -L/mn/stornext/u3/adriaand/ssht/lib/c/ -lssht -L/mn/stornext/u3/adriaand/fftw/lib -lfftw3 -lm 

