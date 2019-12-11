#include "linear_algebra.h"


void testMatrixMemoryFunctions();
void testVectorMemoryFunctions();
void testAlgebraFunctions();
void testMemoryCopyFunctions();

void testMatrixMemoryFunctions(){
	int height = 51;
	int width = 43;
	float max = 200.0
	matrix *d_A;
	matrix *A;
	matrix *B;
	cudaBuildMatrix(d_A, height, width);
	*A = buildMatrix(height, width);
	*B = buildMatrix(height, width);
	randomizeMatrix(A, max);
	copyHostToDevice(A, d_A);
	copyDeviceToHost(d_A, B);
	cudaFreeMatrix(d_A);

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			if(getElement(*A, i, j) != getElement(*B, i, j)){
				printf("failed on i=%d, j=%d", i, j);
			}
		}
	}
	printf("success?");
}


int main(){
	testMatrixMemoryFunctions();
}
