//util
__device__ float getElement(matrix m, int row, int col);
__device__ void setElement(matrix m, int row, int col, float element);
__device__ matrix getSubMatrix(matrix m, int row, int col, int blockSize);

//algebra
__global__ matrixMultiply(matrix a, matrix b, matrix out);
matrix matrixMultiply(matrix a, matrix b);
matrix matrixAdd(matrix a, matrix b);
__global__ matrixAdd(matrix a, matrix b, matrix out);

//memory
matrix cudaBuildMatrix(int height, int width, int stride);
matrix buildMatrix(int height, int width, int stride);
void copyDeviceToHost(matrix *device, matrix *host);
void copyHostToDevice(matrix *host, matrix *device);
void cudaFreeMatrix(matrix *device);
void freeMatrix(matrix *host);
