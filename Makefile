all:  saxpy

saxpy: saxpy.cu
	nvcc -o saxpy saxpy.cu

clean: 
	rm -Rf saxpy