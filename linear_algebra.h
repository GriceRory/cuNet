#include <cuda.h>
#include <cuda_runtime.h>

struct matrix{
	int width;
	int height;
	float *elements;
};

struct vector{
	int length;
	float *elements;
};

#define BLOCK_SIZE 512
#define return_cuda_status if(cudaGetLastError() != cudaSuccess){return cudaPeekAtLastError();}

//util
__device__ float getElement(matrix m, int row, int col);
__device__ void setElement(matrix m, int row, int col, float element);
__device__ float getElement(vector v, int element);
__device__ void setElement(vector v, int element, float value);

//algebra
__global__ void matrixMultiply(vector input, matrix m, vector out);
__global__ void vectorAdd(vector a, vector b, vector out);

//memory
int cudaBuildMatrix(matrix *d_m, int height, int width);
matrix buildMatrix(int height, int width);
int copyDeviceToHost(matrix *device, matrix *host);
int copyHostToDevice(matrix *host, matrix *device);
int cudaFreeMatrix(matrix *device);
void freeMatrix(matrix *host);
vector buildVector(int length);
int cudaBuildVector(vector *v, int length);
int cudaFreeVector(matrix *device);
void freeVector(matrix *host);

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
	d_m->height = cudaMalloc(sizeof(int));
	return_cuda_status
	d_m->width = cudaMalloc(sizeof(int));
	return_cuda_status
	cudaMemcpyHostToDevice(&(d_m->height), &height, sizeof(int), cudaMemcpyDeviceToHost);
	return_cuda_status
	cudaMemcpyHostToDevice(&(d_m->width), &width, sizeof(int), cudaMemcpyDeviceToHost);
	return_cuda_status
	d_m->elements = cudaMalloc(sizeof(float)*height*width);
	return cudaPeekAtLastError();
}
matrix buildMatrix(int height, int width){
	matrix m;
	m.height = height;
	m.width = width;
	m.elements = malloc(sizeof(float)*height*width);
	return m;
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
	v->length= cudaMalloc(sizeof(int));
	return_cuda_status
	cudaMemcpyHostToDevice(&(v->length), &length, sizeof(int), cudaMemcpyDeviceToHost);
	return_cuda_status
	v->elements = cudaMalloc(sizeof(float)*length);
	return cudaPeekAtLastError();
}
int cudaFreeVector(matrix *device){
	cudaFree(&(device->length));
	return_cuda_status
	cudaFree(device->elements);
	return_cuda_status
	cudaFree(device);
	return cudaPeekAtLastError();
}
int freeVector(vector *host){
	free(&(host->length));
	free(host->elements);
	free(*host);
}
int copyDeviceToHost(vector *device, vector *host){
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
