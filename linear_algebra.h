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
__global__ void matrixAdd(matrix target, matrix addition);
__global__ void vectorAdd(vector target, vector addition);

//matrix memory
int cudaBuildMatrix(matrix *d_m, int height, int width);
matrix* buildMatrix(matrix *m, int height, int width);
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
	__shared__ float temp[BLOCK_SIZE];
	if(threadIdx.x > input.length){return;}
	for(int block = 0; block < input.length/BLOCK_SIZE + 1; block++){
		temp[threadIdx.x] = getElement(input, threadIdx.x) * getElement(M, blockIdx.x, threadIdx.x);

		for(int i = 2; i < BLOCK_SIZE; i *= 2){
			__syncthreads();
			if(threadIdx.x < BLOCK_SIZE / i){
				temp[threadIdx.x] += temp[threadIdx.x + BLOCK_SIZE/i];
			}
			__syncthreads();
		}
	}
	setElement(out, blockDim.x*blockIdx.x, temp[0]);
	setElement(out, blockDim.x*blockIdx.x, 1.0);
}

int cudaBuildMatrix(matrix *d_m, int height, int width){
	d_m->height = height;
	d_m->width = width;
	cudaMalloc(&(d_m->elements), sizeof(float)*height*width);
	return cudaGetLastError();
}
matrix* buildMatrix(matrix *m, int height, int width){
	m->height = height;
	m->width = width;
	m->elements = (float*)malloc(sizeof(float)*height*width);
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
	cudaFree(&(device->width));
	return_cuda_status
	cudaFree(&(device->height));
	return_cuda_status
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
	if(device == NULL){return cudaSuccess;}
	cudaFree(&(device->length));
	return_cuda_status
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
	cudaMemcpy(host->elements, device->elements, sizeof(float)*host->length, cudaMemcpyDeviceToHost);
	return cudaGetLastError();
}
int copyHostToDevice(vector *host, vector *device){
	device->length = host->length;
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
