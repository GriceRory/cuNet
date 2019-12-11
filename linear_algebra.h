#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

srand(time(NULL));

struct matrix{
	int width;
	int height;
	float *elements;
};

struct vector{
	int length;
	float *elements;
};

#define BLOCK_SIZE 32
#define return_cuda_status if(cudaPeekAtLastError() != cudaSuccess){return cudaGetLastError();}

//util
__device__ float getElement(matrix m, int row, int col);
__device__ void setElement(matrix m, int row, int col, float element);
__device__ float getElement(vector v, int element);
__device__ void setElement(vector v, int element, float value);
float getElement(matrix m, int row, int col);
void setElement(matrix m, int row, int col, float element);
float getElement(vector v, int element);
void setElement(vector v, int element, float value);

//algebra
__global__ void matrixMultiply(vector input, matrix m, vector out);
__global__ void vectorAdd(vector a, vector b, vector out);
__global__ void matrixAdd(matrix target, matrix addition);
__global__ void vectorAdd(vector target, vector addition);

//matrix memory
int cudaBuildMatrix(matrix *d_m, int height, int width);
matrix* buildMatrix(int height, int width);
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

__device__ float getElement(matrix m, int row, int col){
	return m.elements[row*m.width + col];
}
__device__ void setElement(matrix m, int row, int col, float element){
	m.elements[row*m.width + col] = element;
}
__device__ float getElement(vector v, int element){
	return v.elements[element];
}
__device__ void setElement(vector v, int element, float value){
	v.elements[element] = value;
}

float getElement(matrix m, int row, int col){
	return m.elements[row*m.width + col];
}
void setElement(matrix m, int row, int col, float element){
	m.elements[row*m.width + col] = element;
}
float getElement(vector v, int element){
	return v.elements[element];
}
void setElement(vector v, int element, float value){
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
	float value = getElement(target, idx);
	setElement(target, idx, value + getElement(addition, idx));
}

__global__ void matrixMultiply(vector input, matrix M, vector out){
	__shared__ float temp[BLOCK_SIZE + 1];
	temp[BLOCK_SIZE] = 0
	if(threadIdx.x > input.length || blockDim.x > M.width){return;}

	for(int block = 0; block < input.length/BLOCK_SIZE + 1; block++){
		temp[threadIdx.x] = getElement(input, threadIdx.x) * getElement(M, blockIdx.x, threadIdx.x);

		for(int i = 2; i < BLOCK_SIZE / 2; i *= 2){
			__syncthreads();
			if(threadIdx.x < BLOCK_SIZE / i){
				temp[threadIdx.x] += temp[2*threadIdx.x];
			}
			__syncthreads();
		}
		temp[BLOCK_SIZE] += temp[0];
	}
	setElement(out, blockDim.x, temp[BLOCK_SIZE]);
}
__global__ void vectorAdd(vector a, vector b, vector out){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx > a.length){
		return;
	}

	float Cvalue = getElement(a, idx) + getElement(b, idx);

	setElement(a, idx, Cvalue);
}

int cudaBuildMatrix(matrix *d_m, int height, int width){
	cudaMalloc(&d_m->height, sizeof(int));
	return_cuda_status
	cudaMalloc(d_m->width, sizeof(int));
	return_cuda_status
	cudaMemcpyHostToDevice(&(d_m->height), &height, sizeof(int), cudaMemcpyDeviceToHost);
	return_cuda_status
	cudaMemcpyHostToDevice(&(d_m->width), &width, sizeof(int), cudaMemcpyDeviceToHost);
	return_cuda_status
	cudaMalloc(d_m->elements, sizeof(float)*height*width);
	return cudaPeekAtLastError();
}
matrix* buildMatrix(int height, int width){
	matrix m;
	m.height = height;
	m.width = width;
	malloc(m.elements, sizeof(float)*height*width);
	return &m;
}
int copyDeviceToHost(matrix *device, matrix *host){
	cudaMemcpy(host->width, device->width, sizeof(int), cudaMemcpyDeviceToHost);
	return_cuda_status
	cudaMemcpy(host->height, device->height, sizeof(int), cudaMemcpyDeviceToHost);
	return_cuda_status
	cudaMemcpy(host->elements, device->elements, sizeof(float)*(host->width)*(host->height), cudaMemcpyDeviceToHost);
	return cudaPeekAtLastError();
}
int copyHostToDevice(matrix *host, matrix *device){
	cudaMemcpy(device->width, host->width, sizeof(int), cudaMemcpyHostToDevice);
	return_cuda_status
	cudaMemcpy(device->height, host->height, sizeof(int), cudaMemcpyHostToDevice);
	return_cuda_status
	cudaMemcpy(device->elements, host->elements, sizeof(float)*(host->width)*(host->height), cudaMemcpyHostToDevice);
	return cudaPeekAtLastError();
}
int cudaFreeMatrix(matrix *device){
	cudaFree(&(device->width));
	return_cuda_status
	cudaFree(&(device->height));
	return_cuda_status
	cudaFree(device->elements);
	return_cuda_status
	cudaFree(device);
	return cudaPeekAtLastError();
}
void freeMatrix(matrix *host){
	free(&(host->width));
	free(&(host->height));
	free(host->elements);
	free(host);
}

vector buildVector(int length){
	vector v;
	v.length = length;
	v.elements = malloc(sizeof(float)*length);
	return v;
}
int cudaBuildVector(vector *v, int length){
	cudaMalloc(v->length, sizeof(int));
	return_cuda_status
	cudaMemcpy(&(v->length), &length, sizeof(int), cudaMemcpyDeviceToHost);
	return_cuda_status
	cudaMalloc(v->elements, sizeof(float)*length);
	return cudaPeekAtLastError();
}
int cudaFreeVector(vector *device){
	if(device == NULL){return cudaSuccess;}
	cudaFree(&(device->length));
	return_cuda_status
	cudaFree(device->elements);
	return_cuda_status
	cudaFree(device);
	return cudaPeekAtLastError();
}
void freeVector(vector *host){
	if(host == NULL){return;}
	free(&(host->length));
	free(host->elements);
	free(*host);
	return;
}
int copyDeviceToHost(vector *device, vector *host){
	if(host == null){
		host = malloc(sizeof(vector));
	}else{
		freeVector(host);
		host = malloc(sizeof(vector));
	}
	cudaMemcpy(host->length, device->length, sizeof(int), cudaMemcpyDeviceToHost);
	return_cuda_status
	cudaMemcpy(*host.elements, *device.elements, sizeof(float)*host->length, cudaMemcpyDeviceToHost);
	return cudaPeekAtLastError();
}
int copyHostToDevice(vector *host, vector *device){
	cudaMemcpy(device->length, host->length, sizeof(int), cudaMemcpyHostToDevice);
	return_cuda_status
	cudaMemcpy(*device.elements, *host.elements, sizeof(float)*host->length, cudaMemcpyHostToDevice);
	return cudaPeekAtLastError();
}

void randomizeMatrix(matrix *m, float max){
	for(int i = 0; i < m.height, i++){
		for(int j = 0; j < width, j++){
			setElement(m, j, i, max*((float)rand()/RAND_MAX));
		}
	}
}
void randomizeVector(vector v, float max){
	for(int element = 0; element < v.length; element++){
		setElement(v, element, max*((float)rand()/RAND_MAX));
	}
}
