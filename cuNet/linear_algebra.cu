#include "linear_algebra.h"

__host__ __device__ float get_element(matrix m, int row, int col){
	if(row >= m.height || col >= m.width){
		return NAN;
	}
	return m.elements[row*m.width + col];
}
__host__ __device__ void set_element(matrix m, int row, int col, float element){
	if(row >= m.height || col >= m.width){
		return;
	}
	m.elements[row*m.width + col] = element;
}
__host__ __device__ float get_element(vector v, int element){
	if(v.length <= element){
		return NAN;
	}
	return v.elements[element];;
}
__host__ __device__ void set_element(vector v, int element, float value){
	if(v.length <= element){
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

__global__ void transpose(matrix input, matrix output) {
	if (threadIdx.x + blockIdx.x * blockDim.x >= input.height * input.width) { return; }
	int row = (threadIdx.x + blockIdx.x * blockDim.x) / input.width;
	int col = (threadIdx.x + blockIdx.x * blockDim.x) - input.width * row;
	double value = get_element(input, row, col);
	set_element(output, col, row, value);
}

__global__ void invert(matrix *input, matrix *output) {
	double scalar = 0.0;
	for (int row = 0; row < input->height; ++row) {
		scalar = 1.0/get_element(*input, row, row);
		multiply_element(*input, row, threadIdx.x, scalar);
		multiply_element(*output, row, threadIdx.x, scalar);

		for (int rowAhead = row + 1; rowAhead < input->height; rowAhead++) {
			scalar = get_element(*input, rowAhead, threadIdx.x);
			subtract_element(*input, row, threadIdx.x, rowAhead, threadIdx.x, scalar);
			subtract_element(*output, row, threadIdx.x, rowAhead, threadIdx.x, scalar);
		}		
	}
	for (int row = input->height - 1; row >= 0; --row) {
		for (int rowBehind = row - 1; rowBehind >= 0; --rowBehind) {
			scalar = get_element(*input, rowBehind, threadIdx.x);
			subtract_element(*input, row, threadIdx.x, rowBehind, threadIdx.x, scalar);
			subtract_element(*output, row, threadIdx.x, rowBehind, threadIdx.x, scalar);
		}
	}
}

__device__ void multiply_element(matrix m, int row, int col, double value) {
	double result = value * get_element(m, row, col);
	set_element(m, row, col, value);
}

__device__ void subtract_element(matrix m, int rowFrom, int colFrom, int rowTo, int colTo, double scalar) {
	double result = scalar * (get_element(m, rowTo, colTo) - get_element(m, rowFrom, colFrom));
	set_element(m, rowTo, colTo, result);
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
float dist(vector h_u, vector h_v){
	/*vector *d_v = cuda_build_vector(h_v.length);
	vector *d_u = cuda_build_vector(h_u.length);
	copy_host_to_device(&h_u, d_u);
	copy_host_to_device(d_v, &h_v);

	float *d_distance;
	cudaMalloc(&d_distance, sizeof(float));

	dist<<<1, BLOCK_SIZE>>>(*d_u, *d_v, d_distance);

	float distance = 0;
	cudaMemcpy(&distance, d_distance, sizeof(float), cudaMemcpyDeviceToHost);
	cuda_free_vector(d_v);
	cuda_free_vector(d_u);
	cudaFree(d_distance);
	*/
	float distance = 0;
	for(int element = 0; element < h_u.length; element++){
		distance += (get_element(h_u, element)-get_element(h_v, element))*(get_element(h_u, element)-get_element(h_v, element));
	}

	return distance;
}

__global__ void dist(vector d_u, vector d_v, float *output){
	__shared__ float reduced_sum[BLOCK_SIZE];
	if(threadIdx.x >= BLOCK_SIZE){return;}
	reduced_sum[threadIdx.x] = 0;

	for(int element = threadIdx.x; element < d_u.length; element += BLOCK_SIZE){
		float val = (get_element(d_u, element) - get_element(d_v, element))*(get_element(d_u, element) - get_element(d_v, element));
		if(!isnan(val)){
			reduced_sum[threadIdx.x] += val;
		}
	}

	reduce(reduced_sum);
	if(threadIdx.x == 0){
		*output = reduced_sum[0];
	}
}

__global__ void matrix_multiply(vector d_input, matrix d_M, vector d_out){
	__shared__ float reduced_sum[BLOCK_SIZE];
	if(threadIdx.x >= BLOCK_SIZE){return;}
	reduced_sum[threadIdx.x] = 0;
	int col = blockIdx.x;
	if(col >= d_M.width){return;}

	for(int row = threadIdx.x; row < d_input.length; row += BLOCK_SIZE){
		float val = get_element(d_input, row) * get_element(d_M, row, col);
		if(!isnan(val)){
			reduced_sum[threadIdx.x] += val;
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
/*int copy_device_to_host(matrix *device, matrix *host){
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
}*/

//to do
//this should replace all copy matrix functions.
//should be able to do the same with all other memory copy functions in refactoring.
int copy_matrix(matrix* source, matrix* target, cudaMemcpyKind copy) {
	target->width = source->width;
	target->height = source->height;

	if (copy == cudaMemcpyDeviceToHost || copy == cudaMemcpyHostToHost) {
		free(target->elements);
		target->elements = (float*)malloc(sizeof(float) * target->width * target->height);
	}else if (copy == cudaMemcpyHostToDevice || copy == cudaMemcpyDeviceToDevice) {
		cudaFree(target->elements);
		cudaMalloc(&(target->elements), sizeof(float) * target->width * target->height);
	}else {
		printf("memory copy not a valid enum\n");
	}
	cudaMemcpy(target->elements, source->elements, sizeof(float) * (source->width) * (source->height), copy);
	return cudaGetLastError();
}

int cuda_free_matrix(matrix *device){
	cudaFree(device->elements);
	if(cudaPeekAtLastError() != cudaSuccess){
		return cudaGetLastError();
	}

	//free((void*)device);//there is something wrong with this for some reason.
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
