CC = gcc
NVCC = nvcc

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

DEFS += \
		-DGPU_TIMER \
		$(SPACE)

NVCCFLAGS += -O3 \
			 -use_fast_math \
			 -lm \
			 -lcusolver \
			 $(SPACE)

CFLAGS += -I/usr/local/cuda \
		  -O3 \
		  -Wall \
		  $(SPACE)

# Add source files here
EXECUTABLE  := lud_cuda lud_cusolver
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES     := lud.c lud_cuda.c common.c

OBJS = common.o lud.o lud_kernel.o

.PHONY: all clean 
all : $(EXECUTABLE)

.c.o : 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o $@ -c $<

%.o:	%.cu 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o $@ -c $<


lud_cuda : $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@  $?

lud_cuda : $(OBJS)
	$(NVCC) -D CUSOLVER $(NVCCFLAGS) -o $@  $?

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo

