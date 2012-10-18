all:  waring

waring: waring.cu
	nvcc -o waring waring.cu

clean: 
	rm -Rf waring 
