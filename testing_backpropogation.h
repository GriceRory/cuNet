int test_backpropogation();
int test_train();
int test_backpropogate();
int test_calculate_nodes();
int test_calculate_next_layer_weight_changes();
int test_calculate_next_layer_bias_changes();
int test_calculate_next_layer_node_derivatives();
int test_calculate_node_derivatives();

void initialize_globals();
void free_globals();
float error_term(network d_net, vector h_input, vector expected);


int layers = 3;
float max_weights = 2.0;
float max_biases = 1.0;

int *nodes = (int*)malloc(sizeof(int)*layers);



network h_net;
network d_net;
vector *h_input;
vector *h_expected;
vector *d_input;
vector *d_expected;
vector **node_outputs;
vector **node_derivatives;

int test_train(){
	int failed = 0;
	return failed;
}

int test_backpropogate(){
	int failed = 0;
	printf("testing backpropogate\n\n");
	network weight_and_bias_changes = cuda_build_network(layers, nodes);
	float error_previously = error_term(d_net, *h_input, *h_expected);
	failed |= backpropogate(&d_net, &weight_and_bias_changes, *h_input, h_expected);

	network weight_and_bias_changes_host = build_network(layers, nodes);
	for(int layer = 0; layer < layers; layer++){
		printf("layer %d\nbiases\n", layer);
		print_vector(*weight_and_bias_changes_host.biases[layer]);
		printf("weights\n");
		print_matrix(*weight_and_bias_changes_host.weights[layer]);
	}
	apply_deltas(d_net, weight_and_bias_changes);
	float error_after = error_term(d_net, *h_input, *h_expected);
	if(error_after > error_previously){failed = 1;printf("error was increased\n");}
	free(nodes);
	return failed;
}

int test_calculate_nodes(){
	int failed = 0;
	printf("testing calculateNodes()\n\n");
	vector **node_outputs_host = (vector**)malloc(sizeof(vector*)*h_net.number_of_layers);
	for(int layer = 0; layer < layers; layer++){
		vector *temp = build_vector(10);
		copy_device_to_host(node_outputs[layer], temp);
		node_outputs_host[layer] = temp;
		//print_vector(*temp);
	}
	vector **node_output_host_test = (vector**)malloc(sizeof(vector*)*h_net.number_of_layers);
	node_output_host_test[0] = h_input;
	for(int layer = 1; layer < layers; layer++){
		node_output_host_test[layer] = host_calculate_layer(*h_net.weights[layer-1], *h_net.biases[layer-1], *node_output_host_test[layer-1]);
		print_vector(*node_outputs_host[layer]);
		print_vector(*node_output_host_test[layer]);
		printf("\n\n");
	}

	for(int layer = 0; layer < layers; layer++){
		free_vector(node_output_host_test[layer]);
		free_vector(node_outputs_host[layer]);
	}
	free(node_output_host_test);
	free(node_outputs_host);
	return failed;
}

int test_calculate_next_layer_weight_changes(){
	int failed = 0;
	//int threadsPerBlock = node_outputs[layers-2]->length;
	//int blocks = node_outputs[layers-1]->length;
	//calculate_next_layer_weight_changes<<<threadsPerBlock, blocks>>>(d_net, layers - 1, *node_outputs[layers-1], *node_derivatives[layers-1]);
	return failed;
}

int test_calculate_next_layer_bias_changes(){
	int failed = 0;
	return failed;
}

int test_calculate_next_layer_node_derivatives(){
	int failed = 0;
	printf("testing calculateNextLayerNodeDerivatives()\n\n");

	int threadsPerBlock = h_net.nodes_in_layer[layers - 3];
	int blocks = h_net.nodes_in_layer[layers - 2];
	vector *node_derivatives_next_layer = cuda_build_vector(h_net.nodes_in_layer[layers-1]);
	vector *node_derivatives_this_layer = cuda_build_vector(h_net.nodes_in_layer[layers-2]);
	vector *h_last_layer_derivative = build_vector(d_net.nodes_in_layer[d_net.number_of_layers-1]);

	vector *temp = build_vector(10);
	copy_device_to_host(node_outputs[layers-1], temp);
	for(int node = 0; node < d_net.nodes_in_layer[d_net.number_of_layers-1]; node++){//this is faster on CPU than transferring to a GPU
		float value = 2*(get_element(*temp, node) - get_element(*h_expected, node));
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

void initialize_globals(){
	//initializes global variables for testing
		for(int layer = 0; layer < layers; layer++){nodes[layer] = 10;}
		h_net = build_network(layers, nodes);
		d_net = cuda_build_network(layers, nodes);
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

		node_outputs = calculate_nodes(&d_net, *h_input);
		node_derivatives = calculate_node_derivatives(d_net, node_outputs, d_expected);
}
void free_globals(){
	free_vector(h_input);
	free_vector(h_expected);
	cuda_free_vector(d_input);
	cuda_free_vector(d_expected);

	//free_network(net);
	//cuda_free_network(d_net);
}

int test_backpropogation(){
	printf("testing backpropogation\n");
	initialize_globals();

	int failed = 0;
	failed |= test_calculate_nodes();
	failed |= test_calculate_node_derivatives();
	failed |= test_calculate_next_layer_weight_changes();
	failed |= test_calculate_next_layer_bias_changes();
	failed |= test_calculate_next_layer_node_derivatives();
	failed |= 0;//testBackpropogate();
	failed |= test_train();

	free_globals();
	return failed;
}
