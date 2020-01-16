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
void generate_identity(matrix h_m);
void print_matrix(matrix h_m);
void print_vector(vector h_v);
__host__ __device__ float get_element(matrix m, int row, int col);
__host__ __device__ void set_element(matrix m, int row, int col, float element);
__host__ __device__ float get_element(vector v, int element);
__host__ __device__ void set_element(vector v, int element, float value);

//algebra
__global__ void matrix_multiply(vector d_input, matrix m, vector d_out);
__device__ void reduce(float *reduced_sum);
__global__ void matrix_add(matrix d_target, matrix d_addition);
__global__ void vector_add(vector d_target, vector d_addition);

//matrix memory
matrix* cuda_build_matrix(int height, int width);
matrix* build_matrix(int height, int width);
int copy_device_to_host(matrix *device, matrix *host);
int copy_host_to_device(matrix *host, matrix *device);
int cuda_free_matrix(matrix *device);
void free_matrix(matrix *host);
void randomize_matrix(matrix *h_m, float max);
//vector memory
vector* build_vector(int length);
vector* cuda_build_vector(int length);
int cuda_free_vector(vector *device);
void free_vector(vector *host);
void randomize_vector(vector *h_v, float max);
int copy_host_to_device(vector *host, vector *device);
int copy_device_to_host(vector *device, vector *host);

__host__ __device__ float get_element(matrix m, int row, int col){
	if(row > m.height || col > m.width){
		return 0.0/0.0;
	}
	return m.elements[row*m.width + col];
}
__host__ __device__ void set_element(matrix m, int row, int col, float element){
	if(row > m.height || col > m.width){
		return;
	}
	m.elements[row*m.width + col] = element;
}
__host__ __device__ float get_element(vector v, int element){
	if(v.length < element){
		return 0.0/0.0;
	}
	return v.elements[element];;
}
__host__ __device__ void set_element(vector v, int element, float value){
	if(v.length < element){
		return;
	}
	v.elements[element] = value;
}


__global__ void matrix_add(matrix d_target, matrix d_addition){
	if(threadIdx.x + blockIdx.x*blockDim.x >= d_target.height * d_target.width){return;}
	int row = (threadIdx.x + blockIdx.x*blockDim.x)/d_target.width;
	int col = (threadIdx.x + blockIdx.x*blockDim.x) - d_target.width*row;
	float value = get_element(d_target, row, col) + get_element(d_addition, row, col);
	set_element(d_target, row, col, value);
}
__global__ void vector_add(vector d_target, vector d_addition){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx >= d_target.length){return;}
	float value = get_element(d_target, idx);
	set_element(d_target, idx, value + get_element(d_addition, idx));
}

__global__ void matrix_multiply(vector d_input, matrix d_M, vector d_out){
	__shared__ float reduced_sum[BLOCK_SIZE];
	//if the column(block) is outside the column width.
	int col = blockIdx.x;
	if(col >= d_M.width){return;}
	float vector_value = 0.0;
	float matrix_value = 0.0;

	//for each subset of the vector matrix multiplication of size BLOCK_SIZE
	for(int thread_group = 0; thread_group < (d_input.length/BLOCK_SIZE) + 1; thread_group++){
		//calculate the index we are at in this subset
		int row = threadIdx.x + thread_group*BLOCK_SIZE;
		//if the index is inside the bounds calculate the component
		if(row < d_input.length){
			vector_value = get_element(d_input, row);
			matrix_value = get_element(d_M, row, col);
			reduced_sum[threadIdx.x] += vector_value * matrix_value;
		}
	}
	//calculate the reduced sum of the components in this subset
	reduce(reduced_sum);
	//set the element output.
	if(threadIdx.x == 0){
		set_element(d_out, col, reduced_sum[0]);
	}
}

__device__ void reduce(float *reduced_sum){
	for(int i = 2; i <= BLOCK_SIZE; i *= 2){
		__syncthreads();
		if(threadIdx.x <= (BLOCK_SIZE-1)/i){
			reduced_sum[threadIdx.x] += reduced_sum[threadIdx.x + BLOCK_SIZE/i];
		}
		__syncthreads();
	}
}

matrix* cuda_build_matrix(int height, int width){
	matrix *d_m = (matrix*)malloc(sizeof(matrix*));
	cudaMalloc(&(d_m->elements), sizeof(float)*height*width);
	d_m->height = height;
	d_m->width = width;
	return d_m;
}
matrix* build_matrix(int height, int width){
	matrix *m = (matrix*)malloc(sizeof(matrix));
	m->height = height;
	m->width = width;
	m->elements = (float*)calloc(height*width, sizeof(float));
	return m;
}
int copy_device_to_host(matrix *device, matrix *host){
	host->width = device->width;
	return_cuda_status
	host->height = device->height;
	return_cuda_status
	cudaMemcpy(host->elements, device->elements, sizeof(float)*(host->width)*(host->height), cudaMemcpyDeviceToHost);
	return cudaGetLastError();
}
int copy_host_to_device(matrix *host, matrix *device){
	device->width = host->width;
	return_cuda_status
	device->height = host->height;
	return_cuda_status
	cudaMemcpy(device->elements, host->elements, sizeof(float)*(host->width)*(host->height), cudaMemcpyHostToDevice);
	return cudaGetLastError();
}
int cuda_free_matrix(matrix *device){
	cudaFree(device->elements);
	if(cudaPeekAtLastError() != cudaSuccess){
		return cudaGetLastError();
	}

	free((void*)device);//there is something wrong with this for some reason.
	return cudaGetLastError();
}

void free_matrix(matrix *host){
	free(host->elements);
	free(host);
}

vector* build_vector(int length){
	vector *v = (vector*)(malloc(sizeof(vector)));
	v->length = length;
	v->elements = (float*)calloc(length, sizeof(float));
	return v;
}
vector* cuda_build_vector(int length){
	vector *v = (vector*)malloc(sizeof(vector));
	v->length = length;
	cudaMalloc(&(v->elements), sizeof(float)*length);
	return v;
}
int cuda_free_vector(vector *device){
	cudaFree(device->elements);
	free(device);
	return cudaGetLastError();
}
void free_vector(vector *host){
	if(host == NULL){return;}
	free(host->elements);
	free((void*)host);
	return;
}
int copy_device_to_host(vector *device, vector *host){
	host->length = device->length;
	return_cuda_status
	cudaMemcpy(host->elements, device->elements, sizeof(float)*host->length, cudaMemcpyDeviceToHost);
	return cudaGetLastError();
}
int copy_host_to_device(vector *host, vector *device){
	device->length = host->length;
	return_cuda_status
	cudaMemcpy(device->elements, host->elements, sizeof(float)*host->length, cudaMemcpyHostToDevice);
	return cudaPeekAtLastError();
}

void randomize_matrix(matrix *h_m, float max){
	for(int i = 0; i < h_m->height; i++){
		for(int j = 0; j < h_m->width; j++){
			float val = 2*max*((float)rand()/RAND_MAX) - max;
			set_element(*h_m, i, j, val);
		}
	}
}
void randomize_vector(vector *h_v, float max){
	for(int element = 0; element < h_v->length; element++){
		float val = 2*max*((float)rand()/RAND_MAX) - max;
		set_element(*h_v, element, val);
	}
}

void print_matrix(matrix h_m){
	printf("height = %d, width = %d\n", h_m.height, h_m.width);
	for(int i = 0; i < h_m.height; i++){
		printf("[");
		for(int j = 0; j < h_m.width - 1; j++){
			printf("%3.3f, ", get_element(h_m, i, j));
		}
		printf("%3.3f]\n", get_element(h_m, i, h_m.width - 1));
	}
}

void print_vector(vector h_v){
	printf("[");
	for(int i = 0; i < h_v.length-1; i++){
		printf("%3.3f, ", get_element(h_v, i));
	}
	printf("%3.3f]\n", get_element(h_v, h_v.length - 1));
}

void generate_identity(matrix h_m){
	if(h_m.height != h_m.width){return;}
	for(int i = 0; i < h_m.height; i++){
		set_element(h_m, i, i, 1.0);
	}
}
