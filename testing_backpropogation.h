int test_backpropogation();
int test_train();
int test_backpropogate();
int test_calculate_nodes();
int test_calculate_next_layer_weight_changes();
int test_calculate_next_layer_bias_changes();
int test_calculate_this_layer_node_derivatives();
int test_calculate_node_derivatives();
int test_calculate_last_layer_node_derivatives();

void initialize_globals();
void free_globals();
float error_term(network d_net, vector h_input, vector h_expected);
vector** host_calculate_node_derivatives(network host_net, vector **host_node_outputs, vector *host_expected);
vector** host_calculate_node_outputs(network host_net, vector *host_input);
vector* host_calculate_this_layer_node_derivatives(matrix connecting_weights, vector node_outputs_next_layer, vector node_derivatives_next_layer);
matrix *host_calculate_next_layer_weight_changes(vector* h_node_outputs_next_layer, vector* h_node_outputs_previous_layer, vector* h_node_derivatives_next_layer);
vector* host_calculate_next_layer_bias_changes(vector h_node_outputs, vector h_node_derivatives);

int layers = 20;
float max_weights = 2.0;
float max_biases = 1.0;

int *nodes = (int*)malloc(sizeof(int)*layers);



network h_net;
network d_net;
network d_change;
network h_change;
vector *h_input;
vector *h_expected;
vector *d_input;
vector *d_expected;
vector **d_node_outputs;
vector **d_node_derivatives;
vector **h_node_outputs;
vector **h_node_derivatives;

int test_calculate_nodes(){
	int failed = 0;
 	printf("testing calculateNodes()\n");
	vector **node_output_host_test = host_calculate_node_outputs(h_net, h_input);
	for(int layer = 0; layer < layers; layer++){
		for(int element = 0; element < h_node_outputs[layer]->length; element++){
			float actual = get_element(*h_node_outputs[layer], element);
			float expected = get_element(*node_output_host_test[layer], element);
			if(difference_tollerance(actual, expected, 0.05)){
				failed = 1;
				print_vector(*node_output_host_test[layer]);
				print_vector(*h_node_outputs[layer]);
			}
		}
		if(layer != 0){
			free_vector(node_output_host_test[layer]);
		}
	}
	free(node_output_host_test);
	if(failed){printf("failed in calculate_nodes()\n");}
	return failed;
}

int test_calculate_this_layer_node_derivatives(){
	int failed = 0;
	printf("testing calculate_this_layer_node_derivatives()\n");

	int threadsPerBlock = BLOCK_SIZE;
	int blocks = h_net.nodes_in_layer[layers - 2];
	vector *node_derivatives_this_layer = cuda_build_vector(h_net.weights[0]->height);
	vector *host_node_derivatives_this_layer = build_vector(h_net.weights[0]->width);

	calculate_this_layer_node_derivatves<<<blocks, threadsPerBlock>>>(*d_net.weights[0], *(d_node_outputs[layers-2]), *d_expected, *node_derivatives_this_layer);
	cudaDeviceSynchronize();
	copy_device_to_host(node_derivatives_this_layer, host_node_derivatives_this_layer);


	vector* host_expected_node_derivatives_this_layer = host_calculate_this_layer_node_derivatives(*h_net.weights[0], *h_node_outputs[layers-2], *h_expected);

	for(int element = 0; element < host_expected_node_derivatives_this_layer->length; element++){
		if(difference_tollerance(get_element(*host_expected_node_derivatives_this_layer, element), get_element(*host_node_derivatives_this_layer, element), 0.05)){
			failed = 1;
			printf("host: %f, device %f\n", get_element(*host_expected_node_derivatives_this_layer, element), get_element(*host_node_derivatives_this_layer, element));
		}
	}
	failed |= cudaDeviceSynchronize();
	cuda_free_vector(node_derivatives_this_layer);
	//free_vector(host_node_derivatives_this_layer);
	if(failed){printf("failed in testing calculate_this_layer_node_derivatives\n");}
	return failed;
}

int test_calculate_node_derivatives(){
	printf("testing calculateNodeDerivatives()\n");
	int failed = 0;
	vector **host_node_derivatives_test = host_calculate_node_derivatives(h_net, h_node_outputs, h_expected);

	for(int layer = 0; layer < layers; layer++){
		h_node_derivatives[layer] = build_vector(10);
		copy_device_to_host(d_node_derivatives[layer], h_node_derivatives[layer]);
		for(int element = 0; element < host_node_derivatives_test[layer]->length; element++){
			float expected = get_element(*host_node_derivatives_test[layer], element);
			float actual = get_element(*h_node_derivatives[layer], element);
			if(difference_tollerance(expected, actual, 0.05)){
				failed = 1;
				printf("\nfailed in layer %d\n", layer);
				print_vector(*host_node_derivatives_test[layer]);
				print_vector(*h_node_derivatives[layer]);
			}
		}
	}
	cudaDeviceSynchronize();
	if(failed){printf("failed in calculate_node_derivatives()\n");}
	return failed;
}

int test_calculate_last_layer_node_derivatives(){
	printf("testing calculate_last_layer_node_derivatives()\n");
	vector *difference = cuda_build_vector(10);

	calculate_last_layer_node_derivatives(difference, d_input, d_expected);
	int failed = cudaDeviceSynchronize();
	if(failed){printf("\nfailed kernel execution with cuda status %s\n", cudaGetErrorName((cudaError_t)failed));}
	vector *host_difference = build_vector(h_input->length);
	copy_device_to_host(difference, host_difference);

	for(int element = 0; element < host_difference->length; element++){
		if(difference_tollerance(get_element(*host_difference, element)/2, get_element(*h_input, element) - get_element(*h_expected, element), 0.005)){
			failed = 1;
			printf("element %d, derivative %f, expected %f, actual %f\n", element, get_element(*host_difference, element), get_element(*h_input, element), get_element(*h_expected, element));
		}
	}

	free_vector(host_difference);
	cuda_free_vector(difference);

	if(failed){
		print_vector(*host_difference);
		printf("failed in calculate_last_layer_node_derivatives()\n");
	}
	return failed;
}

int test_calculate_next_layer_weight_changes(){
	printf("testing calculate_next_layer_weight_changes()\n");
	int failed = 0;
	matrix *temp = build_matrix(d_change.weights[0]->height, d_change.weights[0]->width);
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimBlock((d_node_outputs[0]->length/BLOCK_SIZE)+1, (d_node_outputs[1]->length/BLOCK_SIZE)+1);
	calculate_next_layer_weight_changes<<<dimGrid, dimBlock>>>(*d_change.weights[0], *d_node_outputs[0], *d_node_outputs[1], *d_node_derivatives[1]);
	int kernel_execution = cudaDeviceSynchronize();
	if(kernel_execution){printf("failed with error %s\n", cudaGetErrorString((cudaError_t)kernel_execution));return kernel_execution;}
	failed |= kernel_execution;
	copy_device_to_host(d_change.weights[0], temp);
	matrix *test = host_calculate_next_layer_weight_changes(h_node_outputs[0], h_node_outputs[1], h_node_derivatives[1]);

	for(int row = 0; row < d_change.weights[0]->height; row++){
		for(int col = 0; col < d_change.weights[0]->width; col++){
			if(difference_tollerance(get_element(*temp, row, col), get_element(*test, row, col), 0.005)){
				failed = 1;
				printf("failed on element row %d, col %d with GPU %f, CPU %f\n", row, col, get_element(*temp, row, col), get_element(*test, row, col));
			}
		}
	}
	if(failed){
		printf("\nfailed in calculate_next_layer_weight_changes() \n");
		print_matrix(*temp);
		printf("\n");
		print_matrix(*test);
	}
	free_matrix(temp);
	free_matrix(test);
	return failed;
}

matrix* host_calculate_next_layer_weight_changes(vector* h_node_outputs_next_layer, vector* h_node_outputs_previous_layer, vector* h_node_derivatives_next_layer){
	matrix* weight_changes = build_matrix(h_node_outputs_previous_layer->length, h_node_outputs_next_layer->length);
	for(int row = 0; row < weight_changes->height; row++){
		for(int col = 0; col < weight_changes->width; col++){
			float value = sigmoid(get_element(*h_node_outputs_next_layer, col)) * get_element(*h_node_outputs_previous_layer, row) * get_element(*h_node_derivatives_next_layer, col);
			set_element(*weight_changes, row, col, value);
		}
	}
	return weight_changes;
}

int test_calculate_next_layer_bias_changes(){
	printf("testing calculate_next_layer_bias_changes()\n");
	int failed = 0;
	vector *temp = build_vector(d_change.biases[1]->length);
	int threadsPerBlock = BLOCK_SIZE;
	int blocks = d_node_outputs[1]->length/BLOCK_SIZE + 1;
	calculate_next_layer_bias_changes<<<blocks, threadsPerBlock>>>(*d_change.biases[1], *d_node_outputs[1], *d_node_derivatives[2]);
	int kernel_execution = cudaDeviceSynchronize();
	if(kernel_execution){printf("failed with error %s\n", cudaGetErrorString((cudaError_t)kernel_execution));return kernel_execution;}
	failed |= kernel_execution;
	copy_device_to_host(d_change.biases[1], temp);
	vector *test = host_calculate_next_layer_bias_changes(*h_node_outputs[1], *h_node_derivatives[2]);

	for(int element = 0; element < temp->length; element++){
		if(difference_tollerance(get_element(*temp, element), get_element(*test, element), 0.05)){
			failed = 1;
			printf("failed on element element %d, with GPU %f, CPU %f\n", element, get_element(*temp, element), get_element(*test, element));
		}
	}

	if(failed){
		printf("failed in calculate_next_layer_bias_changes()\n");
		print_vector(*temp);
		print_vector(*test);
	}
	free_vector(temp);
	free_vector(test);
	return failed;
}

vector* host_calculate_next_layer_bias_changes(vector h_node_outputs, vector h_node_derivatives){
	vector *bias_change = build_vector(h_node_outputs.length);
	for(int element = 0; element < h_node_derivatives.length; element++){
		float value = get_element(h_node_derivatives, element) * sigmoid(get_element(h_node_outputs, element));
		set_element(*bias_change, element, value);
	}
	return bias_change;
}


int test_train(){
	printf("testing train()\n");
	int failed = 0;
	int dataset_size = 100;
	database *h_sample = build_database(dataset_size);
	database *d_sample = build_database(dataset_size);
	randomize_database(*h_sample, max_biases, max_biases, nodes[0], nodes[layers-1]);
	copy_host_to_device(h_sample, d_sample);
	cudaDeviceSynchronize();
	for(int epoc = 0; epoc < 5; epoc++){
		float errors_before = 0;
		float errors_after = 0;
		for(int element = 0; element < dataset_size; element++){
			errors_before += error_term(d_net, *h_sample->inputs[element], *h_sample->outputs[element]);
		}
		train(&d_net, d_sample, 0.001);
		cudaDeviceSynchronize();
		for(int element = 0; element < dataset_size; element++){
			errors_after += error_term(d_net, *h_sample->inputs[element], *h_sample->outputs[element]);
		}
		if(errors_before < errors_after){
			failed = 1;
			printf("error was increased in element from %f, to %f in epoc %i\n",  errors_before, errors_after, epoc);
		}
	}

	free_database(h_sample);
	cuda_free_database(d_sample);
	return failed;
}

int test_backpropogate(){
	int failed = 0;
	printf("testing backpropogate()\n");
	float error_previously = error_term(d_net, *h_input, *h_expected);
	failed |= backpropogate(&d_net, &d_change, d_input, d_expected);
	copy_device_to_host(&d_net, &h_net);
	copy_device_to_host(&d_change, &h_change);

	apply_deltas(d_net, d_change);
	cudaDeviceSynchronize();
	float error_after = error_term(d_net, *h_input, *h_expected);
	if((error_after-error_previously) / error_previously > 0.4){
		failed = 1;
		//print_network(h_change);
		printf("error was increased by at least 40%% from %f, to %f\n", error_previously, error_after);
	}
	if(failed){printf("failed in backpropogate()\n");}
	return failed;
}

float error_term(network d_net, vector h_input, vector h_expected){
	float error = 0.0;
	vector *h_output = build_vector(d_net.nodes_in_layer[d_net.number_of_layers - 1]);
	run_network(d_net, h_input, h_output);
	for(int element = 0; element < h_expected.length; element++){
		error += (get_element(*h_output, element) - get_element(h_expected, element))*(get_element(*h_output, element) - get_element(h_expected, element));
	}
	return error;
}
void initialize_globals(){
	//initializes global variables for testing
	for(int layer = 0; layer < layers; layer++){nodes[layer] = 200;}
	h_net = build_network(layers, nodes);
	d_net = cuda_build_network(layers, nodes);
	h_change = build_network(layers, nodes);
	d_change = cuda_build_network(layers, nodes);
	h_input = build_vector(nodes[0]);
	h_expected = build_vector(nodes[layers - 1]);
	d_input = cuda_build_vector(nodes[0]);
	d_expected = cuda_build_vector(nodes[layers - 1]);

	randomize_network(h_net, max_weights, max_biases);
	randomize_vector(h_input, max_biases);
	randomize_vector(h_expected, max_biases);

	copy_host_to_device(&h_net, &d_net);
	copy_host_to_device(h_input, d_input);
	copy_host_to_device(h_expected, d_expected);

	d_node_outputs = calculate_nodes(&d_net, d_input);
	h_node_outputs = (vector**)malloc(sizeof(vector*)*h_net.number_of_layers);
	for(int layer = 0; layer < layers; layer++){
		h_node_outputs[layer] = build_vector(10);
		copy_device_to_host(d_node_outputs[layer], h_node_outputs[layer]);
	}
	d_node_derivatives = calculate_node_derivatives(d_net, d_node_outputs, d_expected);
	h_node_derivatives = (vector **)malloc(sizeof(vector*)*h_net.number_of_layers);
	for(int layer = 0; layer < layers; layer++){
		h_node_derivatives[layer] = build_vector(nodes[layer]);
		copy_device_to_host(d_node_derivatives[layer], h_node_derivatives[layer]);
	}
	printf("finished initializing\n");
}

void free_globals(){
	free_vector(h_input);
	free_vector(h_expected);
	cuda_free_vector(d_input);
	cuda_free_vector(d_expected);
}

vector* host_calculate_this_layer_node_derivatives(matrix connecting_weights, vector node_outputs_next_layer, vector node_derivatives_next_layer){
	vector *this_layer_derivatives = build_vector(connecting_weights.width);
	for(int row = 0; row < connecting_weights.height; row++){
		float derivative = 0.0;
		for(int col = 0; col < connecting_weights.width; col++){
			float dE_by_dNodeOutputNextLayer = get_element(node_derivatives_next_layer, col);
			float dNodeOutputNextLayer_by_dNodeInputNextLayer = sigmoid_derivative(get_element(node_outputs_next_layer, col));
			float dNodeInputNextLayer_by_dNodeOutputThisLayerComponent = get_element(connecting_weights, row, col);

			derivative += dE_by_dNodeOutputNextLayer * dNodeOutputNextLayer_by_dNodeInputNextLayer * dNodeInputNextLayer_by_dNodeOutputThisLayerComponent;
		}
		set_element(*this_layer_derivatives, row, derivative);

	}
	return this_layer_derivatives;
}

vector** host_calculate_node_derivatives(network host_net, vector **host_node_outputs, vector *host_expected){
	vector **host_node_derivatives = (vector**)malloc(sizeof(vector*)*host_net.number_of_layers);
	host_node_derivatives[host_net.number_of_layers - 1] = build_vector(host_net.nodes_in_layer[host_net.number_of_layers - 1]);
	for(int node = 0; node < host_net.nodes_in_layer[host_net.number_of_layers - 1];node++){
		float value = -2*(get_element(*host_node_outputs[host_net.number_of_layers - 1], node) - get_element(*host_expected, node));
		set_element(*host_node_derivatives[host_net.number_of_layers - 1], node, value);
	}
	for(int layer = host_net.number_of_layers - 2; layer >= 0; layer--){
		host_node_derivatives[layer] = host_calculate_this_layer_node_derivatives(*host_net.weights[layer], *host_node_outputs[layer+1], *host_node_derivatives[layer+1]);
	}
	return host_node_derivatives;
}

vector** host_calculate_node_outputs(network host_net, vector *host_input){
	vector **node_output_host_test = (vector**)malloc(sizeof(vector*)*host_net.number_of_layers);
	node_output_host_test[0] = host_input;
	for(int layer = 1; layer < host_net.number_of_layers; layer++){
		node_output_host_test[layer] = host_calculate_layer(*host_net.weights[layer-1], *host_net.biases[layer-1], *node_output_host_test[layer-1]);
	}
	return node_output_host_test;
}

int test_backpropogation(){
	printf("testing backpropogation\n");
	initialize_globals();

	int failed = 0;
	failed |= test_calculate_nodes();//done
	failed |= test_calculate_last_layer_node_derivatives();
	failed |= test_calculate_this_layer_node_derivatives();
	failed |= test_calculate_node_derivatives();
	failed |= test_calculate_next_layer_weight_changes();
	failed |= test_calculate_next_layer_bias_changes();
	failed |= test_backpropogate();
	failed |= test_train();

	free_globals();
	return failed;
}
