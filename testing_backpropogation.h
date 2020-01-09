int test_backpropogation();
int test_train();
int test_backpropogate();
int test_calculate_nodes();
int test_calculate_next_layer_weight_changes();
int test_calculate_next_layer_bias_changes();
int test_calculate_next_layer_node_derivatives();
int test_calculate_node_derivatives();
float error_term(network d_net, vector h_input, vector expected);


int layers = 3;
float max_weights = 2.0;
float max_biases = 1.0;

int *nodes = (int*)malloc(sizeof(int)*layers);



network net;
network d_net;
vector *input;
vector *expected;
vector *d_input;
vector *d_expected;
vector **node_outputs;

int test_train(){
	int failed = 0;
	return failed;
}

int test_backpropogate(){
	int failed = 0;
	printf("testing backpropogate\n\n");
	network weight_and_bias_changes = cuda_build_network(layers, nodes);
	float error_previously = error_term(d_net, *input, *expected);
	failed |= backpropogate(&d_net, &weight_and_bias_changes, *input, expected);

	network weight_and_bias_changes_host = build_network(layers, nodes);
	for(int layer = 0; layer < layers; layer++){
		printf("layer %d\nbiases\n", layer);
		print_vector(*weight_and_bias_changes_host.biases[layer]);
		printf("weights\n");
		print_matrix(*weight_and_bias_changes_host.weights[layer]);
	}
	apply_deltas(d_net, weight_and_bias_changes);
	float error_after = error_term(d_net, *input, *expected);
	if(error_after > error_previously){failed = 1;printf("error was increased\n");}
	free(nodes);
	return failed;
}

int test_calculate_nodes(){
	int failed = 0;
	printf("testing calculateNodes()\n\n");

	printf("printing calculate_nodes() output\n");
	vector **node_outputs_host = (vector**)malloc(sizeof(vector*)*net.number_of_layers);
	for(int layer = 0; layer < layers; layer++){
		vector *temp = build_vector(10);
		copy_device_to_host(node_outputs[layer], temp);
		node_outputs_host[layer] = temp;
		print_vector(*temp);
	}


	printf("\nprinting and calculating testing node output values\n\n");
	vector *current_layer_outputs = build_vector(input->length);
	for(int element = 0; element < input->length; element++){
		set_element(*current_layer_outputs, element, get_element(*input, element));
	}
	print_vector(*current_layer_outputs);

	for(int layer = 0; layer < layers; layer++){
		vector *next_layer_outputs = build_vector(net.nodes_in_layer[layer+1]);

		for(int col = 0; col < (net.weights[0])->width; col++){
			float temp = get_element(*(net.biases[layer]), col);
			for(int row = 0; row < (net.weights[0])->height; row++){
				temp += get_element(*(net.weights[0]), row, col) * get_element(*current_layer_outputs, row);
			}
			temp = sigmoid(temp);
			set_element(*next_layer_outputs, col, temp);
		}

		for(int element = 0; element < next_layer_outputs->length; element++){
			if(!(get_element(*node_outputs_host[layer], element) - get_element(*current_layer_outputs, element) < 0.9
					&& get_element(*node_outputs_host[layer], element) - get_element(*current_layer_outputs, element) > -0.9)){
				//printf("failed in layer %d on output element = %d, output = %f, expected = %f\n", layer, element, get_element(*node_outputs_host[layer], element), get_element(*temp_vector, element));
				failed = 1;
			}
		}
		print_vector(*next_layer_outputs);
		current_layer_outputs = next_layer_outputs;
	}
	for(int layer = 0; layer < layers; layer++){
		free_vector(node_outputs_host[layer]);
	}
	free(node_outputs_host);
	return failed;
}

int test_calculate_next_layer_weight_changes(){
	int failed = 0;

	return failed;
}

int test_calculate_next_layer_bias_changes(){
	int failed = 0;
	return failed;
}

int test_calculate_next_layer_node_derivatives(){
	int failed = 0;
	printf("testing calculateNextLayerNodeDerivatives()\n\n");

	int threadsPerBlock = net.nodes_in_layer[layers - 3];
	int blocks = net.nodes_in_layer[layers - 2];
	vector *node_derivatives_next_layer = cuda_build_vector(net.nodes_in_layer[layers-1]);
	vector *node_derivatives_this_layer = cuda_build_vector(net.nodes_in_layer[layers-2]);
	vector *h_last_layer_derivative = build_vector(d_net.nodes_in_layer[d_net.number_of_layers-1]);

	vector *temp = build_vector(10);
	copy_device_to_host(node_outputs[layers-1], temp);
	for(int node = 0; node < d_net.nodes_in_layer[d_net.number_of_layers-1]; node++){//this is faster on CPU than transferring to a GPU
		float value = 2*(get_element(*temp, node) - get_element(*expected, node));
		set_element(*h_last_layer_derivative, node, value);
	}
	copy_host_to_device(h_last_layer_derivative, node_derivatives_next_layer);

	calculate_next_layer_node_derivatves<<<threadsPerBlock, blocks>>>(d_net, layers - 1,  *(node_outputs[layers-1]), *node_derivatives_next_layer, *node_derivatives_this_layer);

	failed |= cudaDeviceSynchronize();
	printf("done testing calculate_next_layer_node_derivatives\n");
	return failed;
}

int test_calculate_node_derivatives(){
	printf("testing calculateNodeDerivatives()\n\n");
	int failed = 0;

	printf("going into calculate derivatives\n\n");
	vector **node_derivatives = calculate_node_derivatives(d_net, node_outputs, d_expected);

	return failed;
}


float error_term(network d_net, vector h_input, vector expected){
	float error = 0.0;
	vector *h_output = build_vector(d_net.nodes_in_layer[d_net.number_of_layers - 1]);
	run_network(d_net, h_input, h_output);
	for(int element = 0; element < h_output->length; element++){
		error += (get_element(*h_output, element) - get_element(expected, element))*(get_element(*h_output, element) - get_element(expected, element));
	}
	return error;
}

int test_backpropogation(){
	printf("testing backpropogation\n");
	//initializes global variables for testing
	for(int layer = 0; layer < layers; layer++){nodes[layer] = 10;}
	net = build_network(layers, nodes);
	d_net = cuda_build_network(layers, nodes);
	input = build_vector(nodes[0]);
	expected = build_vector(nodes[layers - 1]);
	d_input = cuda_build_vector(nodes[0]);
	d_expected = cuda_build_vector(nodes[layers - 1]);

	randomize_network(net, max_weights, max_biases);
	randomize_vector(input, max_biases);
	randomize_vector(expected, max_biases);

	copy_host_to_device(&net, &d_net);
	copy_host_to_device(input, d_input);
	copy_host_to_device(expected, d_expected);

	node_outputs = calculate_nodes(&d_net, *input);

	int failed = 0;//testBackpropogate();
	failed |= test_train();
	failed |= test_calculate_nodes();
	failed |= test_calculate_next_layer_weight_changes();
	failed |= test_calculate_next_layer_bias_changes();
	failed |= test_calculate_next_layer_node_derivatives();
	failed |= test_calculate_node_derivatives();

	free_vector(input);
	free_vector(expected);
	cuda_free_vector(d_input);
	cuda_free_vector(d_expected);

	//free_network(net);
	//cuda_free_network(d_net);

	return failed;
}
