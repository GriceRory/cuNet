#include "linear_algebra.h"

int testMemoryFunctions();
int testMatrixMemoryFunctions(int height, int width, float max);
int testVectorMemoryFunctions(int length, float max);
int testAlgebraFunctions();
int testMatrixMultiply(int height, int width, float max);
int testVectorAdd(int length, float max);
int testMatrixAdd(int height, int width, float max);


int testMemoryFunctions(){
	int matrix_failed = testMatrixMemoryFunctions(10, 11, 20.0);
	int vector_failed = testVectorMemoryFunctions(5, 20.0);
	return matrix_failed || vector_failed;
}
int testMatrixMemoryFunctions(int height, int width, float max){
	int failed = 0;
	printf("testing matrix memory functions\n");
	matrix *d_A = cudaBuildMatrix(height, width);
	matrix *A = buildMatrix(height, width);
	matrix *B = buildMatrix(height, width);
	randomizeMatrix(A, max);

	int copy_A_to_d_A = copyHostToDevice(A, d_A);
	failed = failed || copy_A_to_d_A;
	int copy_d_A_to_B = copyDeviceToHost(d_A, B);
	failed = failed || copy_d_A_to_B;

	int free_d_A = cudaFreeMatrix(d_A);
	failed = failed || free_d_A;
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			if(getElement(*A, i, j) != getElement(*B, i, j)){
				printf("failed on A[%d][%d]=%f, B[%d][%d]=%f\n", i, j, getElement(*A, i, j), i, j, getElement(*B, i, j));
				failed = 1;
			}
		}
	}

	freeMatrix(A);

	if(!failed){
		printf("successfully tested matrix memory functions\n\n\n");
	}else{
		printf("failed = %d, copy to device = %d, copy to host = %d, free_matrix = %d\n", failed, copy_A_to_d_A, copy_d_A_to_B, free_d_A);
		printMatrix(*A);
		printf("\n\n");
		printMatrix(*B);
	}
	return failed;
}

int testVectorMemoryFunctions(int length, float max){
	int failed = 0;
	printf("testing vector memory functions\n");
	vector *d_A = cudaBuildVector(length);;
	vector *A = buildVector(length);;
	vector *B = buildVector(length);;

	randomizeVector(A, max);

	int i = copyHostToDevice(A, d_A);
	int j = copyDeviceToHost(d_A, B);

	cudaFreeVector(d_A);

	for(int i = 0; i < length; i++){
		if(getElement(*A, i) != getElement(*B, i)){
			printf("failed on A[%d]=%f, B[%d]=%f\n", i, getElement(*A, i), i, getElement(*B, i));
			failed = 1;
		}
	}

	if(!failed){printf("successfully tested vector memory functions\n\n\n");
	}else{
		printf("failed\n");
		printVector(*A);
		printf("\n\n");
		printVector(*B);
	}
	return failed;
}

int testAlgebraFunctions(){
	printf("testing algebra functions\n");

	int multiply_failed = testMatrixMultiply(30, 20, 20.0);
	int vector_add_failed = testVectorAdd(1011, 20.0);
	//int matrix_add_failed = testMatrixAdd(100, 100, 20.0);
	return vector_add_failed || multiply_failed;
}
int testMatrixMultiply(int height, int width, float max){
	int failed = 0;
	int length = height;
	printf("\ntesting matrix multiply\n");

	matrix *M = buildMatrix(height, width);;
	matrix *d_M = cudaBuildMatrix(height, width);;
	vector *in = buildVector(length);
	vector *out = buildVector(width);
	vector *d_in = cudaBuildVector(length);
	vector *d_out = cudaBuildVector(width);;

	randomizeMatrix(M, max);
	randomizeVector(in, max);
	int vector_copy_host_to_device = copyHostToDevice(in, d_in);
	int matrix_copy_host_to_device = copyHostToDevice(M, d_M);
	int threads_per_block = BLOCK_SIZE;
	int blocks_per_grid = width;
	cudaDeviceSynchronize();
	matrixMultiply<<<blocks_per_grid, threads_per_block>>>(*d_in, *d_M, *d_out);
	cudaDeviceSynchronize();
	int matrix_multiply = cudaGetLastError();
	int vector_out_copy_device_to_host = copyDeviceToHost(d_out, out);
	int vector_in_copy_device_to_host = copyDeviceToHost(d_in, in);
	int matrix_copy_device_to_host = copyDeviceToHost(d_M, M);

	for(int col = 0; col < width; col++){
		float temp = 0.0;
		for(int row = 0; row < height; row++){
			temp += getElement(*M, row, col) * getElement(*in, row);
		}
		if(getElement(*out, col) - temp > 0.001){
			printf("failed on index %d with out = %.10f, expected = %.10f\n", col, getElement(*out, col), temp);
			failed = 1;
		}
	}
	int free_d_M = cudaFreeMatrix(d_M);
	int free_d_in = cudaFreeVector(d_in);
	int free_d_out = cudaFreeVector(d_out);

	if(failed){
		printf("failed\n");
		printf("in:\n");
		printVector(*in);
		printf("M:\n");
		printMatrix(*M);
		printf("out:\n");
		printVector(*out);
	}else{printf("successfully tested matrix multiplication\n\n\n");}
	return failed;
}
int testVectorAdd(int length, float max){
	int failed = 0;
	printf("testing vector addition\n");
	vector *v = buildVector(length);
	vector *w = buildVector(length);
	vector *u = buildVector(length);
	vector *d_v = cudaBuildVector(length);
	vector *d_w = cudaBuildVector(length);

	randomizeVector(v, max);
	randomizeVector(w, max);

	int copy_v_to_d_v = copyHostToDevice(v, d_v);
	int copy_w_to_d_w = copyHostToDevice(w, d_w);

	int threads_per_block = BLOCK_SIZE;
	int blocks_per_grid = (length / BLOCK_SIZE) + 1;
	vectorAdd<<<threads_per_block, blocks_per_grid>>>(*d_v, *d_w);
	cudaDeviceSynchronize();
	int kernel_execution = cudaGetLastError();
	int copy_d_v_to_u = copyDeviceToHost(d_v, u);


	for(int i = 0; i < length; i++){
		if(getElement(*u, i) - (getElement(*v, i) + getElement(*w, i)) > 0.001){
			failed = 1;
			printf("failed on element %d with u = %.3f != v = %.3f + w = %.3f\n",
					i, getElement(*u, i), getElement(*v, i), getElement(*w, i));
		}
	}

	if(failed){printf("failed\n");
	}else{printf("successfully tested vector addition\n\n\n");}
	return failed;
}
int testMatrixAdd(int height, int width, float max){
	int failed = 0;
	printf("testing matrix addition\n");
	matrix *A = buildMatrix(height, width),
			*B = buildMatrix(height, width),
			*C = buildMatrix(height, width);
	matrix *d_A = cudaBuildMatrix(height, width);
	matrix *d_B = cudaBuildMatrix(height, width);

	randomizeMatrix(A, max);
	randomizeMatrix(B, max);

	int copy_A_to_d_A = copyHostToDevice(A, d_A);
	int copy_B_to_d_B = copyHostToDevice(B, d_B);

	int threads_per_block = BLOCK_SIZE;
	int blocks_per_grid = height*width/BLOCK_SIZE + 1;
	matrixAdd<<<threads_per_block, blocks_per_grid>>>(*d_A, *d_B);
	cudaDeviceSynchronize();
	int kernel_execution = cudaGetLastError();
	int copy_d_A_to_C = copyDeviceToHost(d_A, C);

	for(int i = 0; i < width; i++){
		for(int j = 0; j < height; j++){
			if(getElement(*C, j, i) != getElement(*A, j, i) + getElement(*B, j, i)){
				failed = 1;
				printf("failed on  row %d, col %d with C = %.3f != A = %.3f + B = %.3f\n", i, j, getElement(*C, j, i), getElement(*A, j, i), getElement(*B, j, i));
			}
		}
	}

	if(failed){printf("failed\n");
	}else{printf("successfully tested vector addition\n\n\n");}
	return failed;
}


int testLinearAlgebra(){
	printf("testing linear_algebra.\n\n");
	srand(time(NULL));
	int memory = testMemoryFunctions();
	int algebra = testAlgebraFunctions();
	printf("finished testing linear_algebra.h\n\n\n");
	return memory;// || algebra;
}
