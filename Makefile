
all: cuda-lu cuda-lu-cusolver cuda-lu-cusolver-streaming 

cuda-lu:
	nvcc -O3 -o cuda-lu lud.cu common.c lud_kernel.cu -lcusolver -lcublas

cuda-lu-cusolver:
	nvcc -D CUSOLVER -O3 -o cuda-lu-cusolver lud.cu common.c lud_kernel.cu -lcusolver -lcublas

cuda-lu-cusolver-streaming:
	nvcc -D STREAMING -O3 -o cuda-lu-cusolver-streaming lud.cu common.c lud_kernel.cu -lcusolver -lcublas

clean:
	rm -f cuda-lu cuda-lu-cusolver cuda-lu-cusolver-streaming

