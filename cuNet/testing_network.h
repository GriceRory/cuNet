int test_network();
int test_build_network(int layers);
int test_cuda_build_network(int layers);
int test_randomize_network(int layers);
int test_copy_to_device_functions(int layers);
int test_copy_to_host_functions(int layers);
int test_calculate_layer();
int test_run_network(int layers);

vector* host_run_network(network net, vector input);
vector* host_calculate_layer(matrix weights, vector biases, vector input);


int test_build_network(int layers){
	int failed = 0;
	int *nodes = (int*)malloc(layers*sizeof(int));
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}
	network net = build_network(layers, nodes);
	for(int layer = 0; layer < layers - 1; layer++){
		for(int height = 0; height < net.weights[layer]->height; height++){
			for(int width = 0; width < net.weights[layer]->width; width++){
				failed |= get_weight(net, layer, height, width) != 0.0;
				if(get_weight(net, layer, height, width) != 0.0){printf("failed weight layer = %d, height = %d, width = %d, value = %f\n\n", layer, height, width, get_weight(net, layer, height, width));}
			}
		}
		for(int element = 0; element < net.biases[layer]->length; element++){
			failed |= get_bias(net, layer, element) != 0.0;
			if(get_bias(net, layer, element) != 0.0){printf("failed bias layer = %d, element = %d, value =  %f\n\n", layer, element, get_bias(net, layer, element));}
		}
	}
	if(failed){
		printf("failed testing buildNetwork()\n");
		print_network(net);
	}
	return failed;
}
int test_randomize_network(int layers){
	int failed = 0;
	int *nodes = (int*)malloc(layers * sizeof(int));
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}
	float weightMax = 10.0, biasMax = 20.0;
	network net = build_network(layers, nodes);
	randomize_network(net, weightMax, biasMax);
	for(int layer = 0; layer < layers - 1; layer++){
		for(int height = 0; height < net.weights[layer]->height; height++){
			for(int width = 0; width < net.weights[layer]->width; width++){
				failed |= get_weight(net, layer, height, width) <= -weightMax || get_weight(net, layer, height, width) >= weightMax;
				if(get_weight(net, layer, height, width) <= -weightMax || get_weight(net, layer, height, width) >= weightMax){printf("failed weight layer = %d, height = %d, width = %d, value = %f\n\n", layer, height, width, get_weight(net, layer, height, width));}
			}
		}
		for(int element = 0; element < net.biases[layer]->length; element++){
			failed |= get_bias(net, layer, element) <= -biasMax || get_bias(net, layer, element) >= biasMax;
			if(get_bias(net, layer, element) <= -biasMax || get_bias(net, layer, element) >= biasMax){printf("failed bias layer = %d, element = %d, value =  %f\n\n", layer, element, get_bias(net, layer, element));}
		}
	}
	if(failed){printf("failed testing randomizeNetwork()\n");}
	return failed;
}
int test_cuda_build_network(int layers){
	int failed = 0;
	int* nodes = (int*)malloc(layers * sizeof(int));
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}
	network net = cuda_build_network(layers, nodes);
	failed |= cudaGetLastError();
	if(failed){printf("failed testing randomizeNetwork()\n");}
	return failed;
}
int test_copy_to_device_functions(int layers){
	int failed = 0;
	int* nodes = (int*)malloc(layers * sizeof(int));
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}

	float weightMax = 10.0, biasMax = 20.0;
	network net = build_network(layers, nodes);
	network net_copy = build_network(layers, nodes);
	network net_device = cuda_build_network(layers, nodes);
	randomize_network(net, weightMax, biasMax);
	failed |= copy_host_to_device(&net, &net_device);
	if(failed != cudaSuccess){printf("cudaError on host to device = %d\n", failed);}
	cudaDeviceSynchronize();

	if(failed){printf("\n\nfailed testing copyDeviceToHost() and copyHostToDevice() for networks\n");}
	return failed;
}
int test_copy_to_host_functions(int layers){
	int failed = 0;
	int* nodes = (int*)malloc(layers * sizeof(int));
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}

	float weightMax = 10.0, biasMax = 20.0;
	network net = build_network(layers, nodes);
	network net_copy = build_network(layers, nodes);
	network net_device = cuda_build_network(layers, nodes);
	randomize_network(net, weightMax, biasMax);
	failed |= copy_host_to_device(&net, &net_device);
	if(failed != cudaSuccess){printf("cudaError on host to device = %d\n", failed);}
	cudaDeviceSynchronize();
	failed |= copy_device_to_host(&net_device, &net_copy);
	cudaDeviceSynchronize();

	for(int layer = 0; layer < layers - 1; layer++){
		if(net.nodes_in_layer[layer] != net_copy.nodes_in_layer[layer]){
			printf("failed on nodes in layer, net = %d, net_copy = %d\n", net.nodes_in_layer[layer], net_copy.nodes_in_layer[layer]);
		}
		if(net.weights[layer]->height != net_copy.weights[layer]->height || net.weights[layer]->width != net_copy.weights[layer]->width){
			printf("failed on matrix height or with net, net_copy height = %d, %d, width = %d, %d\n\n", net.weights[layer]->height, net_copy.weights[layer]->height, net.weights[layer]->width, net_copy.weights[layer]->width);
		}
		for(int row = 0; row < net.weights[layer]->height; row++){
			for(int col = 0; col < net.weights[layer]->width; col++){
				if(get_weight(net, layer, row, col) != get_weight(net_copy, layer, row, col)){
					failed = 1;
					printf("failed on weight %f != %f, layer = %d, row = %d, col = %d\n", get_weight(net, layer, row, col), get_weight(net_copy, layer, row, col), layer, row, col);
				}
			}
		}
		for(int element = 0; element < net.biases[layer]->length; element++){
			if(get_bias(net, layer, element) != get_bias(net_copy, layer, element)){
				failed = 1;
				printf("failed on bias %f != %f, layer = %d, element = %d\n", get_bias(net, layer, element), get_bias(net_copy, layer, element), layer, element);
			}
		}
	}

	if(failed){printf("\n\nfailed testing copyDeviceToHost() and copyHostToDevice() for networks\n");}
	return failed;
}
int test_calculate_layer(){
	int failed = 0;
	float weightMax = 0.5, biasMax = 0.5;
	int height = 65, width = 65;
	matrix *weights = build_matrix(height, width);
	matrix *d_weights = cuda_build_matrix(height, width);

	vector *input = build_vector(height),
			*output = build_vector(width),
			*biases = build_vector(width),
			*d_input = cuda_build_vector(height),
			*d_output = cuda_build_vector(width),
			*d_biases = cuda_build_vector(width),
			*output_test = build_vector(width);

	randomize_matrix(weights, weightMax);
	randomize_vector(input, biasMax);
	randomize_vector(biases, biasMax);

	copy_host_to_device(weights, d_weights);
	copy_host_to_device(input, d_input);
	copy_host_to_device(biases, d_biases);

	cudaDeviceSynchronize();
	calculate_layer(*d_weights, *d_biases, *d_input, *d_output, streams[0]);
	cudaDeviceSynchronize();
	copy_device_to_host(d_output, output_test);
	cudaDeviceSynchronize();

	output = host_calculate_layer(*weights, *biases, *input);

	for(int col = 0; col < weights->width; col++){
		if(difference_tollerance(get_element(*output_test, col), get_element(*output_test, col), 0.001)){

			printf("failed on index %d with out = %.10f, expected = %.10f\n",
					col, get_element(*output_test, col), get_element(*output_test, col));
			failed = 1;
		}
	}
	if(failed){printf("\n\nfailed testing calculateLayer()\n");}
	return failed;
}
int test_run_network(int layers){
	int failed = 0;
	int* nodes = (int*)malloc(layers * sizeof(int));
	for(int i = 0; i < layers; i++){
		nodes[i] = 500;
	}

	float weightMax = 0.1, biasMax = 0.2;
	network net = build_network(layers, nodes);
	randomize_network(net, weightMax, biasMax);
	network net_device = cuda_build_network(layers, nodes);
	int error = copy_host_to_device(&net, &net_device);
	if(error != cudaSuccess){printf("error copy host to device = %d\n\n", error);}
	vector *input = build_vector(net.nodes_in_layer[0]);
	randomize_vector(input, biasMax);
	vector *output = build_vector(net.nodes_in_layer[1]);
	error = run_network(net_device, *input, output, streams[0]);
	if(error != cudaSuccess){printf("error run network = %d\n\n", error);}

	vector *output_test = host_run_network(net, *input);

	for(int element = 0; element < output->length; element++){
		if(difference_tollerance(get_element(*output, element), get_element(*output_test, element), 0.0001)){
			printf("failed on output element = %d, output = %f, expected = %f\n", element, get_element(*output, element), get_element(*output_test, element));
			failed = 1;
		}
	}
	if(failed){printf("failed testing runNetwork()\n");}
	return failed;
}

vector* host_run_network(network net, vector input){
	vector *output = &input;
	for(int layer = 0; layer < net.number_of_layers - 1; layer++){
		vector *temp =  host_calculate_layer(*net.weights[layer], *net.biases[layer], *output);
		output = temp;
	}
	return output;
}
vector* host_calculate_layer(matrix weights, vector biases, vector input){
	vector *output = build_vector(weights.width);
	for(int col = 0; col < weights.width; col++){
		float temp = get_element(biases, col);
		for(int row = 0; row < weights.height; row++){
			temp += get_element(input, row) * get_element(weights, row, col);
		}
		set_element(*output, col, sigmoid(temp));
	}
	return output;
}

int test_network(){
	int failed = 0;
	int network_size = 10;
	failed |= test_build_network(network_size);
	failed |= test_cuda_build_network(network_size);
	failed |= test_randomize_network(network_size);
	failed |= test_copy_to_device_functions(network_size);
	failed |= test_copy_to_host_functions(network_size);
	failed |= test_calculate_layer();
	failed |= test_run_network(network_size);
	printf("Finished testing network.h\n");
	return failed;
}
