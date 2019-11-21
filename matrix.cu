struct matrix{
	int width;
	int height;
	int stride;
	float *elements;
};

#include "matrix.h"

#define BLOCK_SIZE 32

//util
__device__ float getElement(matrix m, int row, int col){
	return m.elements[row*m.stride + col];
}
__device__ void setElement(matrix m, int row, int col, float element){
	m.elements[row*m.stride + col] = element;
}
__device__ matrix getSubmatrix(matrix m, int row, int col){
	matrix sub;
	sub.width = BLOCK_SIZE;
	sub.height = BLOCK_SIZE;
	sub.stride = m.stride;
	sub.elements = &m.elements[m.stride*BLOCK_SIZE*row + BLOCK_SIZE*col];
	return sub;
}

//algebra

__global__ void matrixMultiply(matrix A, matrix B, matrix out){
	int blockRow = blockIdx.x;
	int blockCol = blockIdx.y;

	matrix outSub = getSubmatrix(c, blockRow, blockCol, BLOCK_SIZE);
	float Cvalue = 0.0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for(int i = 0; i < (a.width / BLOCK_SIZE); i++){
		matrix Asub = getSubmatrix(A, blockRow, i);
		matrix Bsub = getSubmatrix(B, i, blockCol);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = getElement(Asub, row, col);
		Bs[row][col] = getElement(Bsub, row, col);

		__syncthreads();
		for(int j = 0; j < BLOCK_SIZE; j++){
			Cvalue += As[row][j]*Bs[j][col];
		}
		__syncthreads();
	}
	setElement(out, row, col, Cvalue);
}


__global__ void matrixAdd(matrix a, matrix b){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if(row >= a.height || col >= a.width){
		return;
	}

	Cvalue = getElement(a, row, col) + getElement(b, row, col);

	setElement(a, row, col, Cvalue);
}

//memory
void cudaBuildMatrix(matrix *d_m, int height, int width, int stride){
	*d_m.height = cudaMalloc(sizeof(int));
	*d_m.width = cudaMalloc(sizeof(int));
	*d_m.stride = cudaMalloc(sizeof(int));
	*d_m.height = height;
	*d_m.width = width;
	*d_m.stride = stride;
	*d_m.elements = cudaMalloc(sizeof(float)*height*width);
}
matrix buildMatrix(int height, int width, int stride){
	struct matrix m;
	m.height = height;
	m.width = width;
	m.stride = stride;
	m.elements = malloc(sizeof(float)*height*width);
	return m;
}
void copyDeviceToHost(matrix *device, matrix *host){
	cudaMemcpy(*host.width, *device.width, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*host.height, *device.height, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*host.stride, *device.stride, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*host.elements, *device.elements, sizeof(float)*(*host.width)*(*host.height), cudaMemcpyDeviceToHost);
}
void copyHostToDevice(matrix *host, matrix *device){
	cudaMemcpy(*device.width, *host.width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(*device.height, *host.height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(*device.stride, *host.stride, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(*device.elements, *host.elements, sizeof(float)*(*host.width)*(*host.height), cudaMemcpyHostToDevice);
}
void cudaFreeMatrix(matrix *device){
	cudaFree(&(*device.width));
	cudaFree(&(*device.height));
	cudaFree(&(*device.stride));
	cudaFree(*device.elements);
	cudaFree(device);
}
void freeMatrix(matrix *host){
	Free(&(*host.width));
	Free(&(*host.height));
	Free(&(*host.stride));
	Free(*host.elements);
	Free(host);
}
