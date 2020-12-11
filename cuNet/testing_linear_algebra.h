int test_memory_functions();
int test_matrix_memory_functions(int height, int width, float max);
int test_vector_memory_functions(int length, float max);
int test_algebra_functions();
int test_matrix_multiply(int height, int width, float max);
int test_vector_add(int length, float max);
int test_matrix_add(int height, int width, float max);


int test_memory_functions(){
	int matrix_failed = 0;
	int vector_failed = 0;
	for(int dimentions = 60; dimentions < 80; dimentions++){
		matrix_failed = test_matrix_memory_functions(dimentions, dimentions, 20.0);
		vector_failed = test_vector_memory_functions(dimentions, 20.0);
		for(int height = 60; height < 100; height++){
			matrix_failed |= test_matrix_memory_functions(height, dimentions, 20.0);
		}
	}
	return matrix_failed || vector_failed;
}
int test_matrix_memory_functions(int height, int width, float max){
	int failed = 0;
	matrix *d_A = cuda_build_matrix(height, width);
	matrix *A = build_matrix(height, width);
	matrix *B = build_matrix(height, width);
	randomize_matrix(A, max);
	
	int copy_A_to_d_A = copy_matrix(A, d_A, cudaMemcpyHostToDevice);
	failed = failed || copy_A_to_d_A;
	int copy_d_A_to_B = copy_matrix(d_A, B, cudaMemcpyDeviceToHost);
	failed = failed || copy_d_A_to_B;
	int free_d_A = cuda_free_matrix(d_A);
	failed = failed || free_d_A;
	
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			if(get_element(*A, i, j) != get_element(*B, i, j)){
				printf("failed on A[%d][%d]=%f, B[%d][%d]=%f\n", i, j, get_element(*A, i, j), i, j, get_element(*B, i, j));
				failed = 1;
			}
		}
	}
	
	if(failed){
		printf("failed = %d, copy to device = %d, copy to host = %d, free_matrix = %d\n", failed, copy_A_to_d_A, copy_d_A_to_B, free_d_A);
		print_matrix(*A);
		printf("\n\n");
		print_matrix(*B);
	}
	free_matrix(A);
	free_matrix(B);

	return failed;
}
int test_vector_memory_functions(int length, float max){
	int failed = 0;
	vector *d_A = cuda_build_vector(length);;
	vector *A = build_vector(length);;
	vector *B = build_vector(length);;

	randomize_vector(A, max);

	failed |= copy_vector(A, d_A, cudaMemcpyHostToDevice);
	failed |= copy_vector(d_A, B, cudaMemcpyDeviceToHost);

	failed |= cuda_free_vector(d_A);

	for(int i = 0; i < length; i++){
		if(get_element(*A, i) != get_element(*B, i)){
			printf("failed on A[%d]=%f, B[%d]=%f\n", i, get_element(*A, i), i, get_element(*B, i));
			failed = 1;
		}
	}

	if(failed){
		printf("failed\n");
		print_vector(*A);
		printf("\n\n");
		print_vector(*B);
	}
	free(A);
	free(B);
	return failed;
}

int test_algebra_functions(){
	int multiply_failed = test_matrix_multiply(30, 20, 20.0);
	int matrix_add_failed = test_matrix_add(100, 100, 20.0);
	int vector_add_failed = test_vector_add(1011, 20.0);

	for(int dimentions = 62; dimentions < 65; dimentions++){
		vector_add_failed |= test_vector_add(dimentions, 20.0);
		multiply_failed |= test_matrix_multiply(dimentions, dimentions, 2.0);
		matrix_add_failed |= test_matrix_add(dimentions, dimentions, 20.0);
	}
	return matrix_add_failed || vector_add_failed || multiply_failed;
}
int test_matrix_multiply(int height, int width, float max){

	int failed = 0;
	int length = height;

	matrix *M = build_matrix(height, width),
			*d_M = cuda_build_matrix(height, width);
	vector *in = build_vector(length),
			*out = build_vector(width),
			*d_in = cuda_build_vector(length),
			*d_out = cuda_build_vector(width);

	randomize_matrix(M, max);
	randomize_vector(in, max);
	int vector_copy_host_to_device = copy_vector(in, d_in, cudaMemcpyHostToDevice);
	int matrix_copy_host_to_device = copy_matrix(M, d_M, cudaMemcpyHostToDevice);
	int threads_per_block = BLOCK_SIZE;
	int blocks_per_grid = width;
	matrix_multiply<<<blocks_per_grid, threads_per_block>>>(*d_in, *d_M, *d_out);
	cudaDeviceSynchronize();
	int matrix_multiply = cudaGetLastError();
	int vector_out_copy_device_to_host = copy_vector(d_out, out, cudaMemcpyDeviceToHost);
	int vector_in_copy_device_to_host = copy_vector(d_in, in, cudaMemcpyDeviceToHost);
	int matrix_copy_device_to_host = copy_matrix(d_M, M, cudaMemcpyDeviceToHost);
	for(int col = 0; col < width; col++){
		float temp = 0.0;
		for(int row = 0; row < height; row++){
			temp += get_element(*M, row, col) * get_element(*in, row);
		}
		if(get_element(*out, col) - temp > 1 || get_element(*out, col) - temp < -1){
			printf("failed on index %d with out = %.10f, expected = %.10f\n", col, get_element(*out, col), temp);
			failed = 1;
		}
	}
	int free_d_M = cuda_free_matrix(d_M);
	int free_d_in = cuda_free_vector(d_in);
	int free_d_out = cuda_free_vector(d_out);

	if(failed){
		printf("failed matrix multiply\n\n\n\n");
	}
	free(M);
	free(in);
	free(out);

	return failed;
}
int test_vector_add(int length, float max){
	int failed = 0;
	vector *v = build_vector(length),
			*w = build_vector(length),
			*u = build_vector(length),
			*d_v = cuda_build_vector(length),
			*d_w = cuda_build_vector(length);

	randomize_vector(v, max);
	randomize_vector(w, max);

	int copy_v_to_d_v = copy_vector(v, d_v, cudaMemcpyHostToDevice);
	int copy_w_to_d_w = copy_vector(w, d_w, cudaMemcpyHostToDevice);

	int threads_per_block = BLOCK_SIZE;
	int blocks_per_grid = (length / BLOCK_SIZE) + 1;
	vector_add<<<blocks_per_grid, threads_per_block>>>(*d_v, *d_w);
	cudaDeviceSynchronize();
	int kernel_execution = cudaGetLastError();
	int copy_d_v_to_u = copy_vector(d_v, u, cudaMemcpyDeviceToHost);


	for(int i = 0; i < length; i++){
		if(get_element(*u, i) - (get_element(*v, i) + get_element(*w, i)) > 0.001){
			failed = 1;
			printf("failed on element %d with u = %.3f != v = %.3f + w = %.3f\n",
					i, get_element(*u, i), get_element(*v, i), get_element(*w, i));
		}
	}

	cuda_free_vector(d_v);
	cuda_free_vector(d_w);

	if(failed){printf("failed\n");}

	free_vector(v);
	free_vector(w);
	free_vector(u);
	return failed;
}
int test_matrix_add(int height, int width, float max){
	int failed = 0;
	matrix *A = build_matrix(height, width),
			   *B = build_matrix(height, width),
			   *C = build_matrix(height, width),
				 *d_A = cuda_build_matrix(height, width),
				 *d_B = cuda_build_matrix(height, width);

	randomize_matrix(A, max);
	randomize_matrix(B, max);

	int copy_A_to_d_A = copy_matrix(A, d_A, cudaMemcpyHostToDevice);
	int copy_B_to_d_B = copy_matrix(B, d_B, cudaMemcpyHostToDevice);

	int threads_per_block = BLOCK_SIZE;
	int blocks_per_grid = height*width/BLOCK_SIZE + 1;
	matrix_add<<<blocks_per_grid, threads_per_block>>>(*d_A, *d_B);
	cudaDeviceSynchronize();
	int kernel_execution = cudaGetLastError();
	int copy_d_A_to_C = copy_matrix(d_A, C, cudaMemcpyDeviceToHost);

	for(int i = 0; i < width; i++){
		for(int j = 0; j < height; j++){
			if(get_element(*C, j, i) != get_element(*A, j, i) + get_element(*B, j, i)){
				failed = 1;
				printf("failed on  row %d, col %d with C = %.3f != A = %.3f + B = %.3f\n", i, j, get_element(*C, j, i), get_element(*A, j, i), get_element(*B, j, i));
			}
		}
	}

	if(failed){printf("failed\n");}
	free_matrix(A);
	free_matrix(B);
	free_matrix(C);
	cuda_free_matrix(d_A);
	cuda_free_matrix(d_B);
	return failed;
}


int test_linear_algebra(){
	printf("testing linear_algebra.\n\n");
	int memory = test_memory_functions();
	int algebra = test_algebra_functions();
	printf("finished testing linear_algebra.h\n");
	return memory || algebra;
}
