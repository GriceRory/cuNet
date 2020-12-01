#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
	int width;
	int height;
	float *elements;
}matrix;

typedef struct{
	int length;
	float *elements;
}vector;

#define BLOCK_SIZE 32
#define return_cuda_status if(cudaPeekAtLastError() != cudaSuccess){return cudaGetLastError();}



//util
void generate_identity(matrix h_m);
void print_matrix(matrix h_m);
void print_vector(vector h_v);
__host__ __device__ float get_element(matrix m, int row, int col);
__host__ __device__ void set_element(matrix m, int row, int col, float element);
__host__ __device__ float get_element(vector v, int element);
__host__ __device__ void set_element(vector v, int element, float value);

//vector algebra
__global__ void vector_add(vector d_target, vector d_addition);
__global__ void vector_subtract(vector d_output, vector d_value, vector d_subtraction);
__global__ void scalar_multiply(vector d_vector, float scalar);
int equals(vector u, vector v);
float dist(vector h_u, vector h_v);
__global__ void dist(vector d_u, vector d_v, float *output);

//matrix algebra
__global__ void matrix_multiply(vector d_input, matrix m, vector d_out);
__device__ void reduce(float *reduced_sum);
__global__ void matrix_add(matrix d_target, matrix d_addition);
__global__ void scalar_multiply(matrix d_matrix, float scalar);
__global__ void transpose(matrix input, matrix output);
__global__ void invert(matrix input, matrix output);
__device__ void multiply_element(matrix m, int row, int col, double value);
__device__ void subtract_element(matrix m, int rowFrom, int colFrom, int rowTo, int colTo, double scalar);


//matrix memory
matrix* cuda_build_identity_matrix();
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

