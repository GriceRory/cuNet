#include "linear_algebra.h"

void testMemoryFunctions();
void testMatrixMemoryFunctions(int height, int width, float max);
void testVectorMemoryFunctions(int length, float max);
void testAlgebraFunctions();
void testMatrixMultiply(int height, int width, float max);
void testVectorAdd(int length, float max);
void testMatrixAdd(int height, int width, float max);


void testMemoryFunctions(){
	testMatrixMemoryFunctions(51, 50, 20.0);
	testVectorMemoryFunctions(21, 20.0);
}
void testMatrixMemoryFunctions(int height, int width, float max){
	int failed = 0;
	printf("testing matrix memory functions\n");
	matrix d_A;
	matrix A;
	matrix B;
	cudaBuildMatrix(&d_A, height, width);
	A = buildMatrix(height, width);
	B = buildMatrix(height, width);
	randomizeMatrix(&A, max);

	int i = copyHostToDevice(&A, &d_A);
	int j = copyDeviceToHost(&d_A, &B);

	int k = cudaFreeMatrix(&d_A);

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			if(getElement(A, i, j) != getElement(B, i, j)){
				printf("failed on A[%d][%d]=%f, B[%d][%d]=%f\n", i, j, getElement(A, i, j), i, j, getElement(B, i, j));
				failed = 1;
			}
		}
	}


	if(!failed || i != cudaSuccess || j != cudaSuccess){
		printf("successfully tested matrix memory functions\n\n\n");
	}else{
		printf("failed, i = %d, j = %d\n", i, j);
		printMatrix(A);
		printf("\n\n");
		printMatrix(B);
	}
}
void testVectorMemoryFunctions(int length, float max){
	int failed = 0;
	printf("testing vector memory functions\n");
	vector d_A;
	vector A;
	vector B;
	cudaBuildVector(&d_A, length);
	A = buildVector(length);
	B = buildVector(length);
	randomizeVector(A, max);

	int i = copyHostToDevice(&A, &d_A);
	int j = copyDeviceToHost(&d_A, &B);

	cudaFreeVector(&d_A);

	for(int i = 0; i < length; i++){
		if(getElement(A, i) != getElement(B, i)){
			printf("failed on A[%d]=%f, B[%d]=%f\n", i, getElement(A, i), i, getElement(B, i));
			failed = 1;
		}
	}

	if(!failed){printf("successfully tested vector memory functions\n\n\n");
	}else{
		printf("failed\n");
		printVector(A);
		printf("\n\n");
		printVector(B);
	}
}

void testAlgebraFunctions(){
	printf("testing algebra functions\n");

	testMatrixMultiply(30, 30, 20.0);
	testVectorAdd(1011, 20.0);
	testMatrixAdd(53, 56, 20.0);
}
void testMatrixMultiply(int height, int width, float max){
	int failed = 0;
	int length = height;
	printf("\ntesting matrix multiply\n");

	matrix M;
	matrix d_M;
	vector in, out;
	vector d_in, d_out;
	M = buildMatrix(height, width);
	int build_d_M = cudaBuildMatrix(&d_M, height, width);
	in = buildVector(length);
	out = buildVector(width);
	int build_d_in = cudaBuildVector(&d_in, length);
	int build_d_out = cudaBuildVector(&d_out, width);
	randomizeMatrix(&M, max);
	randomizeVector(in, max);
	int vector_copy_host_to_device = copyHostToDevice(&in, &d_in);
	int matrix_copy_host_to_device = copyHostToDevice(&M, &d_M);
	int threads_per_block = BLOCK_SIZE;
	int blocks = width;
	matrixMultiply<<<threads_per_block, blocks>>>(d_in, d_M, d_out);
	cudaDeviceSynchronize();
	int matrix_multiply = cudaGetLastError();
	printf("cuda last error matrix multiply: %d\n", matrix_multiply);
	int vector_copy_device_to_host = copyDeviceToHost(&d_out, &out);

	for(int i = 0; i < width; i++){
		float temp = 0.0;
		for(int j = 0; j < height; j++){
			temp += getElement(M, j, i) * getElement(in, j);
		}
		if(getElement(out, i) != temp){
			printf("failed on index %d with out = %.3f, expected = %.3f\n", i, getElement(out, i), temp);
			failed = 1;
		}
	}
	cudaFreeMatrix(&d_M);
	cudaFreeVector(&d_in);
	cudaFreeVector(&d_out);

	if(failed){
		printf("failed\n");
		printf("in:\n");
		printVector(in);
		printf("M:\n");
		printMatrix(M);
		printf("out:\n");
		printVector(out);
	}else{printf("successfully tested matrix multiplication\n\n\n");}
}
void testVectorAdd(int length, float max){
	int failed = 0;
	printf("testing vector addition\n");
	vector v = buildVector(length);
	vector w = buildVector(length);
	vector u = buildVector(length);
	vector d_v, d_w;
	cudaBuildVector(&d_v, length);
	cudaBuildVector(&d_w, length);

	randomizeVector(v, max);
	randomizeVector(w, max);

	copyHostToDevice(&v, &d_v);
	copyHostToDevice(&w, &d_w);

	int threads_per_block = BLOCK_SIZE;
	int blocks = (length / BLOCK_SIZE) + 1;
	vectorAdd<<<threads_per_block, blocks>>>(d_v, d_w);
	cudaDeviceSynchronize();
	copyDeviceToHost(&d_v, &u);



	for(int i = 0; i < length; i++){
		if(getElement(u, i) != getElement(v, i) + getElement(w, i)){
			failed = 1;
			printf("failed on element %d with u = %.3f != v = %.3f + w = %.3f\n", i, getElement(u, i), getElement(v, i), getElement(w, i));
		}
	}

	if(failed){printf("failed\n");
	}else{printf("successfully tested vector addition\n\n\n");}
}
void testMatrixAdd(int height, int width, float max){
	int failed = 0;
	printf("testing matrix addition\n");
	matrix A = buildMatrix(height, width), B = buildMatrix(height, width), C = buildMatrix(height, width);
	matrix d_A, d_B;
	cudaBuildMatrix(&d_A, height, width);
	cudaBuildMatrix(&d_B, height, width);

	randomizeMatrix(&A, max);
	randomizeMatrix(&B, max);

	copyHostToDevice(&A, &d_A);
	copyHostToDevice(&B, &d_B);

	int threads_per_block = BLOCK_SIZE;
	int blocks = height*width/BLOCK_SIZE + 1;
	matrixAdd<<<threads_per_block, blocks>>>(d_A, d_B);
	cudaDeviceSynchronize();
	copyDeviceToHost(&d_A, &C);

	for(int i = 0; i < width; i++){
		for(int j = 0; j < height; j++){
			if(getElement(C, j, i) != getElement(A, j, i) + getElement(B, j, i)){
				failed = 1;
				printf("failed on  row %d, col %d with C = %.3f != A = %.3f + B = %.3f\n", i, j, getElement(C, j, i), getElement(A, j, i), getElement(B, j, i));
			}
		}
	}

	if(failed){printf("failed\n");
	}else{printf("successfully tested vector addition\n\n\n");}
}


void testLinearAlgebra(){
	printf("testing linear_algebra.\n\n");
	srand(time(NULL));
	testMemoryFunctions();
	testAlgebraFunctions();
	printf("finished testing linear_algebra.h\n\n\n");
}
