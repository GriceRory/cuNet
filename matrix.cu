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
