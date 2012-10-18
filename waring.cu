// CUDA programming example
// A program that computes Y = A*X + Y where X and Y are vectors
// of real numbers and A is a scalar real number.
// Inspired by the Dan Ernst, Brandon Holt CUDA Programming Model talk

#include "book.h"
#include <set>

__global__ void waring_cuda(int size, int cap, int *n, int *f, int *v) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int r;
    if(i < size) {
        for (int j=0; j<size; j++) {
            // obtain new result
            r =  (v[i] + v[j]) % cap;
            f[r] = 1; 
        }
    }
}

int gf_pow(int n, int m, int p) {
    int res = n;
    for (int i=1; i<m; i++) {
        res = (res * n) % p;
    }
    return res;
}

int main(int argc, char* argv[]) {   
    if ( (argc < 3) || (atoi(argv[1])==1 && argc < 5) ) {
        printf("Usage: %s <0=CPU or 1=GPU> <N> <Blocks> <Threads/Block>\n",argv[0]);
        return 1; 
    }

    std::set<int> S;
    int list_size;
    int *v; int  *f;             // Pointers to host arrays
    int *v_dev, *f_dev; // Pointers to device arrays

    int N = atoi(argv[2]);
    size_t size = N * sizeof(int); // Compute size of arrays in bytes

    v = (int *)malloc(size);            // Allocate array on host
    f = (int *)malloc(size);            // Allocate array on host

    HANDLE_ERROR(cudaMalloc ((void**) &f_dev, size));
    HANDLE_ERROR(cudaMalloc ((void**) &v_dev, size));
    


    // Initialize host array

    for (int i=0; i<N; i++) {
        f[i] = 0; v[i] = 0; 
    }
    int t;
    for (int i=0; i<N;i++) {
        t = gf_pow(i,11,N);
        f[t] = 1;
      	S.insert(t); 
    }
    std::set<int>::iterator it = S.begin();
    int k = 0;
    for (;it!=S.end();it++) {
       v[k] = *it;
       k++;
    }
    list_size = S.size();
    for (k=0;k<list_size;k++) 
        printf("%d ",v[k]);
    printf("\n");
    
    if (atoi(argv[1])==1) {

        unsigned int n_blocks = atoi(argv[3]);
        unsigned int block_size = atoi(argv[4]);

        // Copy host array to the GPU
        HANDLE_ERROR(cudaMemcpy(f_dev, f, size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(v_dev, v, size, cudaMemcpyHostToDevice));
        
        // Do calculation on device:
        //printf("Deploying %d blocks with %d threads per block\n", n_blocks, block_size);
        waring_cuda <<< n_blocks, block_size >>> (list_size, N, &N, f_dev, v_dev);
        

        
        // Retrieve result from device and store it in host array
        cudaMemcpy(f, f_dev, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(v, v_dev, size, cudaMemcpyDeviceToHost);
    }
/*
    else {
        saxpy(N, 10.0, x, y);
    }
*/
    // Print results if you have the patience :-)
    for (int i=0; i<N; i++) 
    	printf("%d + %d \n", f[i], v[i]);

    for (int i=0; i<N; i++) 
    	if(f[i]) printf("%d ", i); 


    // I don't so I'll just print the last result
    //printf("%f : %f \n", x[N-1], y[N-1]);
    
    // Cleanup
    free(f); free(v); 
    cudaFree(f_dev); cudaFree(v_dev); 
}
