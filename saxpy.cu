// CUDA programming example
// A program that computes Y = A*X + Y where X and Y are vectors
// of real numbers and A is a scalar real number.
// Inspired by the Dan Ernst, Brandon Holt CUDA Programming Model talk

#include "book.h"

__global__ void saxpy_cuda(int n, float a, float *x, float *y) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < n) {
        y[i] = a*x[i] + y[i];
    }
}

void saxpy(int n, float a, float *x, float *y) {
	for (int i=0; i<n; i++) {
		y[i] = a*x[i] + y[i];
    }
}

int main(int argc, char* argv[]) {   
    if ( (argc < 3) || (atoi(argv[1])==1 && argc < 5) ) {
        printf("Usage: %s <0=CPU or 1=GPU> <N> <Blocks> <Threads/Block>\n",argv[0]);
        return 1; 
    }


    float *x; float *y;             // Pointers to host arrays
    float *x_dev, *y_dev; // Pointers to device arrays

    int N = atoi(argv[2]);
    size_t size = N * sizeof(int); // Compute size of arrays in bytes

    x = (float *)malloc(size);            // Allocate array on host
    y = (float *)malloc(size);            // Allocate array on host

    HANDLE_ERROR(cudaMalloc ((void**) &x_dev, size));
    HANDLE_ERROR(cudaMalloc ((void**) &y_dev, size));
    


    // Initialize host array
    for (int i=0; i<N; i++) {
    	x[i] = i * 1.0 ; y[i] = i * 1.0 + 1;
    }
    
    if (atoi(argv[1])==1) {

        unsigned int n_blocks = atoi(argv[3]);
        unsigned int block_size = atoi(argv[4]);

        // Copy host array to the GPU
        HANDLE_ERROR(cudaMemcpy(x_dev, x, size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(y_dev, y, size, cudaMemcpyHostToDevice));
        
        // Do calculation on device:
        //printf("Deploying %d blocks with %d threads per block\n", n_blocks, block_size);
        saxpy_cuda <<< n_blocks, block_size >>> (N, 10.0, x_dev, y_dev);
        
        // Retrieve result from device and store it in host array
        cudaMemcpy(y, y_dev, size, cudaMemcpyDeviceToHost);
    }
    else {
        saxpy(N, 10.0, x, y);
    }
    // Print results if you have the patience :-)
    //for (int i=0; i<N; i++) 
    //	printf("%d + %d = %d \n", a[i], b[i], c[i]);

    // I don't so I'll just print the last result
    printf("%f : %f \n", x[N-1], y[N-1]);
    
    // Cleanup
    free(x); free(y); 
    cudaFree(x_dev); cudaFree(y_dev); 
}