struct matrix{
	int width;
	int height;
	int stride;
	float *elements;
};

#include <matrix.h>

//util
__device__ float getElement(matrix m, int row, int col){
	return m.elements[row*m.stride + col];
}
__device__ void setElement(matrix m, int row, int col, float element){
	m.elements[row*m.stride + col] = element;
}
__device__ matrix getSubmatrix(matrix m, int row, int col, int blockSize){
	matrix sub;
	sub.width = blockSize;
	sub.height = blockSize;
	sub.stride = m.stride;
	sub.elements = &m.elements[m.stride*blockSize*row + blockSize*col];
	return sub;
}

//algebra

__global__ void matrixMultiply(matrix A, matrix B, matrix *out, int blockSize){
	int blockRow = blockIdx.x;
	iny blockCol = blockIdx.y;

	matrix outSub = getSubmatrix(c, blockRow, blockCol, blockSize);
	float Cvalue = 0.0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for(int i = 0; i < (a.width / blockSize); i++){
		matrix Asub = getSubmatrix(A, blockRow, i, blockSize);
		matrix Bsub = getSubmatrix(B, i, blockCol, blockSize);

		__shared__ float As[blockSize][blockSize];
		__shared__ float Bs[blockSize][blockSize];

		As[row][col] = getElement(Asub, row, col);
		Bs[row][col] = getElement(Bsub, row, col);

		__syncthreads();
		for(int j = 0; j < blockSize; j++){
			Cvalue += As[row][j]*Bs[j][col];
		}
		__syncthreads();
	}
	setElement(*out, row, col, Cvalue);
}
matrix matrixAdd(matrix a, matrix b){
	matrix out;
	matrix d_A = cudaMalloc(sizeof(struct matrix));
	matrix d_B = cudaMalloc(sizeof(struct matrix));
	matrix d_out = cudaMalloc(sizeof(struct matrix));

	cudaBuildMatrix(&d_A, a.height, a.width, a.stride);
	cudaBuildMatrix(&d_B, b.height, b.width, b.stride);
	cudaBuildMatrix(&d_out, b.height, b.width, b.stride);

	copyHostToDevice(a, d_A);
	copyHostToDevice(b, d_B);

	matrixAdd<<<1, 1, 1>>>(d_A, d_B, d_out);

	copyDeviceToHost(d_out, out);
	cudaFreeMatrix(d_out);
	cudaFreeMatrix(d_A);
	cudaFreeMatrix(d_B);
	return out;
}
__global__ void matrixAdd(matrix a, matrix b, matrix *out){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if(row >= a.height || col >= a.width){
		return;
	}

	Cvalue = getElement(A, row, col) + getElement(B, row, col);

	setElement(*out, row, col, Cvalue);
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
