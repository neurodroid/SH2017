#! /bin/bash

# LIBFFTW=`pkg-config --libs fftw3 fftw3f`
# CFLAGSFFTW=`pkg-config --cflags fftw3 fftw3f`
LIBFFTW="-L${MKLROOT}/lib/intel64 -mkl -lmkl_core -lmkl_intel_lp64"
CFLAGSFFTW="-I${MKLROOT}/include"
# CC="clang -Wno-dangling-else"
# LD="clang"
CC=icc
LD=icc
OPTFLAGS="-m64 -Ofast -ipo -march=native -funroll-loops"
# OPTFLAGS="-O0 -g3"
LDOPTFLAGS=${OPTFLAGS}

# rm burak-${USER} gc_dynamics *.o

echo "${CC} ${OPTFLAGS} -std=gnu99 ${CFLAGSFFTW} -c entorhinal5.c -o entorhinal.o"
${CC} ${OPTFLAGS} -std=gnu99 ${CFLAGSFFTW} -c entorhinal5.c -o entorhinal.o
echo "${LD} entorhinal.o ${LDOPTFLAGS} ${LIBFFTW} -o burak-${USER}"
${LD} entorhinal.o ${LDOPTFLAGS} ${LIBFFTW} -o burak-${USER}

echo "${CC} ${OPTFLAGS} -std=gnu99 ${CFLAGSFFTW} -c gc_dynamics.c -o gc_dynamics.o"
${CC} ${OPTFLAGS} -std=gnu99 ${CFLAGSFFTW} -c gc_dynamics.c -o gc_dynamics.o
echo "${LD} gc_dynamics.o ${LDOPTFLAGS} ${LIBFFTW} -o gc_dynamics-${USER}"
${LD} gc_dynamics.o ${LDOPTFLAGS} ${LIBFFTW} -o gc_dynamics-${USER}
