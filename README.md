# cuda-lu

This directory contains 3 applications that are all implementations of LU decomposition: the ``cuda-lu`` for the Rodinia Suite implementation, the ``cuda-lu-cusolver`` for the cuSolver implementation, and the ``cuda-lu-cusolver-streaming`` for the cuSolver with stream implementation.

To compile, type ``make all``.

To run and profile these applications with NVPROF, first generate the input using the gen_input application in the ``tools`` folder. For example, to generate a 1024x1024 matrix input, 
```bash
cd tools && gen_input 1024
```

And then, run and profile these applications

```bash
nvprof ./cuda-lu -i tools/1024.dat > 1024.out

nvprof ./cuda-lu-cusolver -i tools/1024.dat > 1024.out

nvprof ./cuda-lu-cusolver-streaming -i tools/1024.dat > 1024.out
```
