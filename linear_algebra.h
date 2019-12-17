#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct{
	int width;
	int height;
	float *elements;
}matrix;

typedef struct{
	int length;
	float *elements;
}vector;

#define BLOCK_SIZE 64
#define return_cuda_status if(cudaPeekAtLastError() != cudaSuccess){return cudaGetLastError();}



//util
void generateIdentity(matrix m);
void printMatrix(matrix m);
void printVector(vector v);
__host__ __device__ float getElement(matrix m, int row, int col);
__host__ __device__ void setElement(matrix m, int row, int col, float element);
__host__ __device__ float getElement(vector v, int element);
__host__ __device__ void setElement(vector v, int element, float value);

//algebra
__global__ void matrixMultiply(vector input, matrix m, vector out);
__device__ void reduce(float *reduced_sum);
__global__ void matrixAdd(matrix target, matrix addition);
__global__ void vectorAdd(vector target, vector addition);

//matrix memory
int cudaBuildMatrix(matrix *d_m, int height, int width);
matrix buildMatrix(int height, int width);
int copyDeviceToHost(matrix *device, matrix *host);
int copyHostToDevice(matrix *host, matrix *device);
int cudaFreeMatrix(matrix *device);
void freeMatrix(matrix *host);
void randomizeMatrix(matrix *m, float max);
//vector memory
vector buildVector(int length);
int cudaBuildVector(vector *v, int length);
int cudaFreeVector(vector *device);
void freeVector(vector *host);
void randomizeVector(vector v, float max);
int copyHostToDevice(vector *host, vector *device);
int copyDeviceToHost(vector *device, vector *host);

__host__ __device__ float getElement(matrix m, int row, int col){
	return m.elements[row*m.width + col];
}
__host__ __device__ void setElement(matrix m, int row, int col, float element){
	m.elements[row*m.width + col] = element;
}
__host__ __device__ float getElement(vector v, int element){
	return v.elements[element];
}
__host__ __device__ void setElement(vector v, int element, float value){
	v.elements[element] = value;
}


__global__ void matrixAdd(matrix target, matrix addition){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.x + blockIdx.x*blockDim.x;
	float value = getElement(target, x, y);
	setElement(target, x, y, value + getElement(addition, x, y));
}
__global__ void vectorAdd(vector target, vector addition){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < target.length){return;}
	float value = getElement(target, idx);
	setElement(target, idx, value + getElement(addition, idx));
}

__global__ void matrixMultiply(vector input, matrix M, vector out){
	__shared__ float reduced_sum[BLOCK_SIZE + 1];
	//if the column(block) is outside the column width.
	int col = blockIdx.x;
	if(col > M.width){return;}

	//for each subset of the vector matrix multiplication of size BLOCK_SIZE
	for(int block = 0; block < input.length/BLOCK_SIZE + 1; block++){
		//calculate the index we are at in this subset
		int row = threadIdx.x + block*BLOCK_SIZE;
		//if the index is past the length of the vector
		if(row > input.length){
			reduced_sum[threadIdx.x] = 0;
			return;
		}
		//calculate this component of the multiplication
		reduced_sum[threadIdx.x] = getElement(input, row) * getElement(M, row, col);

		//calculate the reduced sum of the components in this subset
		reduce(reduced_sum);

		//add the reduced sum of the components in this subset to the tally of all the reduced sums.
		reduced_sum[BLOCK_SIZE] += reduced_sum[0];
	}
	//set the element output.
	setElement(out, blockDim.x*blockIdx.x, reduced_sum[BLOCK_SIZE]);
}

__device__ void reduce(float *reduced_sum){
	for(int i = 2; i < BLOCK_SIZE; i *= 2){
		__syncthreads();
		if(threadIdx.x < BLOCK_SIZE / i){
			reduced_sum[threadIdx.x] += reduced_sum[threadIdx.x + BLOCK_SIZE/i];
		}
		__syncthreads();
	}
}

int cudaBuildMatrix(matrix *d_m, int height, int width){
	d_m->height = height;
	d_m->width = width;
	cudaMalloc(&(d_m->elements), sizeof(float)*height*width);
	return cudaGetLastError();
}
matrix buildMatrix(int height, int width){
	matrix m;
	m.height = height;
	m.width = width;
	m.elements = (float*)malloc(sizeof(float)*height*width);
	return m;
}
int copyDeviceToHost(matrix *device, matrix *host){
	host->width = device->width;
	return_cuda_status
	host->height = device->height;
	return_cuda_status
	cudaMemcpy(host->elements, device->elements, sizeof(float)*(host->width)*(host->height), cudaMemcpyDeviceToHost);
	return cudaGetLastError();
}
int copyHostToDevice(matrix *host, matrix *device){
	device->width = host->width;
	return_cuda_status
	device->height = host->height;
	return_cuda_status
	cudaMemcpy(device->elements, host->elements, sizeof(float)*(host->width)*(host->height), cudaMemcpyHostToDevice);
	return cudaGetLastError();
}
int cudaFreeMatrix(matrix *device){
	cudaFree(device->elements);
	return_cuda_status
	cudaFree(device);
	return cudaGetLastError();
}
void freeMatrix(matrix *host){
	free(host->elements);
	free(host);
}

vector buildVector(int length){
	vector v;
	v.length = length;
	v.elements = (float*)(malloc(sizeof(float)*length));
	return v;
}
int cudaBuildVector(vector *v, int length){
	v->length = length;
	cudaMalloc(&(v->elements), sizeof(float)*length);
	return cudaGetLastError();
}
int cudaFreeVector(vector *device){
	cudaFree(device->elements);
	return_cuda_status
	cudaFree(device);
	return cudaGetLastError();
}
void freeVector(vector *host){
	if(host == NULL){return;}
	free(&(host->length));
	free(host->elements);
	free(host);
	return;
}
int copyDeviceToHost(vector *device, vector *host){
	host->length = device->length;
	return_cuda_status
	cudaMemcpy(host->elements, device->elements, sizeof(float)*host->length, cudaMemcpyDeviceToHost);
	return cudaGetLastError();
}
int copyHostToDevice(vector *host, vector *device){
	device->length = host->length;
	return_cuda_status
	cudaMemcpy(device->elements, host->elements, sizeof(float)*host->length, cudaMemcpyHostToDevice);
	return cudaPeekAtLastError();
}

void randomizeMatrix(matrix *m, float max){
	for(int i = 0; i < m->height; i++){
		for(int j = 0; j < m->width; j++){
			float r = max*((float)rand()/RAND_MAX) - max/2.0;
			setElement(*m, j, i, r);
		}
	}
}
void randomizeVector(vector v, float max){
	for(int element = 0; element < v.length; element++){
		setElement(v, element, max*((float)rand()/RAND_MAX) - max/2.0);
	}
}

void printMatrix(matrix m){
	for(int i = 0; i < m.width; i++){
		printf("[");
		for(int j = 0; j < m.height - 1; j++){
			printf("%3.3f, ", getElement(m, i, j));
		}
		printf("%3.3f]\n", getElement(m, i, m.height-1));
	}
}

void printVector(vector v){
	printf("[");
	for(int i = 0; i < v.length-1; i++){
		printf("%3.3f, ", getElement(v, i));
	}
	printf("%3.3f]\n", getElement(v, v.length - 1));
}

void generateIdentity(matrix m){
	if(m.height != m.width){return;}
	for(int i = 0; i < m.height; i++){
		setElement(m, i, i, 1.0);
	}
}
