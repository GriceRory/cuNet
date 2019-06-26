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
__device__ matrix getSubMatrix(matrix m, int row, int col, int blockSize){
	matrix sub;
	sub.width = blockSize;
	sub.height = blockSize;
	sub.stride = m.stride;
	sub.elements = &m.elements[m.stride*blockSize*row + blockSize*col];
	return sub;
}


//algebra
__global__ matrixMultiply(matrix a, matrix b, matrix out){
	int blockRow = blockIdx.x;
	int blockCol = blockIdx.y;
	int blockSize = blockDim.x;

	matrix outSub = getSubMatrix(out, blockRow, blockCol, blockSize);

	float Cval = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for(int i = 0; i < (a.width/blockSize); i++){
		matrix Asub = getSubMatrix(a, blockRow, i, blockSize);
		matrix Bsub = getSubMatrix(b, blockRow, i, blockSize);

		__shared__ float As[blockSize][blockSize];
		__shared__ float Bs[blockSize][blockSize];

		__syncthreads();

		for(int j = 0; j < blockSize; j++){
			Cval += As[row][j] * Bs[j][col];
		}
		__syncthreads();
	}
	setElement(outSub, row, col, Cval);
}
matrix matrixMultiply(matrix a, matrix b){
	matrix d_a;
	copyHostToDevice(&a, &d_a);
	matrix d_b;
	copyHostToDevice(&b, &d_b);
	matrix d_out = cudaBuildMatrix(a.height, b.width, a.stride);

	int blockSize = 16
	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(b.width/dimBlock.x, a.height/dimBlock.y);
	matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_b, d_out);

	matrix out;
	copyDeviceToHost(&d_out, &out);
	return out;
}
matrix matrixAdd(matrix a, matrix b){
	matrix d_a;
	copyHostToDevice(&a, &d_a);
	matrix d_b;
	copyHostToDevice(&b, &d_b);
	matrix d_out = cudaBuildMatrix(a.height, b.width, a.stride);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(b.width/threadsPerBlock.x, a.height/threadsPerBlock.y);
	matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_b, d_out);

	matrix out;
	copyDeviceToHost(&d_out, &out);
	return out;
}
__global__ matrixAdd(matrix a, matrix b, matrix out){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if(row*a.width + col < a.width*a.height){
		float val = getElement(a, row, col) + getElement(b, row, col);
		setElement(out, row, col, val);
	}
}

//memory
matrix cudaBuildMatrix(int height, int width, int stride){
	matrix d_m;
	d_m.height = height;
	d_m.width = width;
	d_m.stride = stride;
	cudaMalloc(&d_m.elements, sizeof(float)*height*width);
	return d_m;
}
matrix cudaBuildMatrix(int height, int width, int stride){
	matrix m;
	m.height = height;
	m.width = width;
	m.stride = stride;
	Malloc(&m.elements, sizeof(float)*height*width);
	return m;
}
void copyDeviceToHost(matrix *device, matrix *host){
	*host.width = *device.width;
	*host.height = *device.height;
	*host.stride = *device.stride;
	size_t size = (*device.height)*(*device.width)*sizeof(float);
	malloc(&(*host.elements)), size);
	cudaMemcpy(host, device, size, cudaMemcpyHostToDevice);
}
void copyHostToDevice(matrix *host, matrix *device){
	*device.width = *host.width;
	*device.height = *host.height;
	*device.stride = *host.stride;
	size_t size = (*device.height)*(*device.width)*sizeof(float);
	cudaMalloc(&(*device.elements)), size);
	cudaMemcpy(deice, host, size, cudaMemcpyHostToDevice);
}
void cudaFreeMatrix(struct matrix *device){
  cudaFree(&(*device.height));
  cudaFree(&(*device.width));
  cudaFree(&(*device.stride));
  cudaFree(&(*device.elements));
  cudaFree(*device);
}
void freeMatrix(matrix *host){
  cudaFree(&(*host.height));
  cudaFree(&(*host.width));
  cudaFree(&(*host.stride));
  cudaFree(&(*host.elements));
  cudaFree(*host);
}
