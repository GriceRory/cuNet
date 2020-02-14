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
__global__ void vector_subtract(vector d_output, vector d_value, vector d_subtraction);
__global__ void scalar_multiply(vector d_vector, float scalar);
__global__ void scalar_multiply(matrix d_matrix, float scalar);
int equals(vector u, vector v);
float dist(vector u, vector v);

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

__global__ void vector_subtract(vector d_output, vector d_value, vector d_subtraction){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx >= d_output.length){return;}
	float value = get_element(d_value, idx) - get_element(d_subtraction, idx);
	set_element(d_output, idx, value);
}

__global__ void scalar_multiply(vector d_vector, float scalar){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx >= d_vector.length){return;}
	float value = get_element(d_vector, idx) * scalar;
	set_element(d_vector, idx, value);
}

__global__ void scalar_multiply(matrix d_matrix, float scalar){
	if(threadIdx.x + blockIdx.x*blockDim.x >= d_matrix.height * d_matrix.width){return;}
	int row = (threadIdx.x + blockIdx.x*blockDim.x)/d_matrix.width;
	int col = (threadIdx.x + blockIdx.x*blockDim.x) - d_matrix.width*row;
	if(row >= d_matrix.height || col >= d_matrix.width){return;}
	float value = get_element(d_matrix, row, col) * scalar;
	set_element(d_matrix, row, col, value);
}

int equals(vector u, vector v){
	if(u.length != v.length){return 0;}
	for(int i = 0; i < u.length; i++){
		if(get_element(u, i) != get_element(v, i)){
			return 0;
		}
	}
	return 1;
}

//this could be made into a device function for a little bit of extra speed.
float dist(vector u, vector v){
	float distance = 0;
	for(int i = 0; i < u.length; i++){
		distance += get_element(u, i) * get_element(v, i);
	}
	return distance;
}

__global__ void matrix_multiply(vector d_input, matrix d_M, vector d_out){
	__shared__ float reduced_sum[BLOCK_SIZE];
	reduced_sum[threadIdx.x] = 0;
	int col = blockIdx.x;
	if(col >= d_M.width){return;}

	for(int row = threadIdx.x; row < d_input.length; row += BLOCK_SIZE){
		reduced_sum[threadIdx.x] += get_element(d_input, row) * get_element(d_M, row, col);
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
	if(device->width != host->width || device->height != host->height){
		host->width = device->width;
		host->height = device->height;
		free(host->elements);
		host->elements = (float*)malloc(sizeof(float) * device->width * device->height);
	}
	cudaMemcpy(host->elements, device->elements, sizeof(float)*(host->width)*(host->height), cudaMemcpyDeviceToHost);
	return cudaGetLastError();
}
int copy_host_to_device(matrix *host, matrix *device){
	if(device->width != host->width || device->height != host->height){
		device->width = host->width;
		device->height = host->height;
		cudaFree(device->elements);
		cudaMalloc(&(device->elements), sizeof(float) * device->width * device->height);
	}
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
	if(host->length != device->length){
		host->length = device->length;
		free(host->elements);
		host->elements = (float*)malloc(sizeof(float)*host->length);
	}
	cudaMemcpy(host->elements, device->elements, sizeof(float)*host->length, cudaMemcpyDeviceToHost);
	return cudaGetLastError();
}
int copy_host_to_device(vector *host, vector *device){
	if(device->length != host->length){
		device->length = host->length;
		cudaFree(device->elements);
		cudaMalloc(&(device->elements), sizeof(float)*device->length);
	}

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
	printf("length = %d\n[", h_v.length);
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
