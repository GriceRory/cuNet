int testBackpropogation();
int testTrain();
int testBackpropogate();
int testCalculateNodes();
int testCalculateNextLayerWeightChanges();
int testCalculateNextLayerBiasChanges();
int testCalculateNextLayerNodeDerivatives();
int testCalculateNodeDerivatives();


int layers = 5;
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

int testTrain(){
	int failed = 0;
	return failed;
}
int testBackpropogate(){
	int failed = 0;
	printf("testing backpropogate\n\n");
	network weight_and_bias_changes = cudaBuildNetwork(layers, nodes);
	failed |= backpropogate(&d_net, &weight_and_bias_changes, *input, *expected);

	network weight_and_bias_changes_host = buildNetwork(layers, nodes);
	for(int layer = 0; layer < layers; layer++){
		printf("layer %d\nbiases\n", layer);
		printVector(*weight_and_bias_changes_host.biases[layer]);
		printf("weights\n");
		printMatrix(*weight_and_bias_changes_host.weights[layer]);
	}

	free(nodes);
	return failed;
}
int testCalculateNodes(){
	int failed = 0;
	printf("testing calculateNodes()\n\n");
	node_outputs = calculateNodes(&d_net, *input);

	vector **node_outputs_host = (vector**)malloc(sizeof(vector*)*net.number_of_layers);
	for(int layer = 0; layer < layers; layer++){
		vector *temp = buildVector(10);
		copyDeviceToHost(node_outputs[layer], temp);
		node_outputs_host[layer] = temp;
	}

	vector *temp_vector = buildVector(input->length);
	for(int element = 0; element < input->length; element++){
		setElement(*temp_vector, element, getElement(*input, element));
	}

	for(int layer = 0; layer < layers; layer++){
		vector *nextLayer = buildVector(net.nodes_in_layer[layer+1]);
		for(int col = 0; col < (net.weights[0])->width; col++){
			float temp = 0.0;
			for(int row = 0; row < (net.weights[0])->height; row++){
				temp += getElement(*(net.weights[0]), row, col) * getElement(*temp_vector, row);
			}
			temp += getElement(*(net.biases[layer]), col);
			temp = sigmoid(temp);
			setElement(*nextLayer, col, temp);
		}
		for(int element = 0; element < nextLayer->length; element++){
			if(!(getElement(*node_outputs_host[layer], element) - getElement(*temp_vector, element) < 0.9 && getElement(*node_outputs_host[layer], element) - getElement(*temp_vector, element) > -0.9)){
				printf("failed in layer %d on output element = %d, output = %f, expected = %f\n", layer, element, getElement(*node_outputs_host[layer], element), getElement(*temp_vector, element));
				failed = 1;
			}
		}
		temp_vector = nextLayer;
	}
	for(int layer = 0; layer < layers; layer++){
		cudaFreeVector(node_outputs[layer]);
		freeVector(node_outputs_host[layer]);
	}
	free(node_outputs);
	free(node_outputs_host);
	return failed;
}
int testCalculateNextLayerWeightChanges(){
	int failed = 0;
	return failed;
}
int testCalculateNextLayerBiasChanges(){
	int failed = 0;
	return failed;
}
int testCalculateNextLayerNodeDerivatives(){
	int failed = 0;
	printf("testing calculateNextLayerNodeDerivatives()\n\n");
	//vector **node_outputs = calculateNodes(&d_net, *d_input);

	int threadsPerBlock = net.nodes_in_layer[layers - 3];
	int blocks = net.nodes_in_layer[layers - 2];

	vector *node_derivatives_next_layer = cudaBuildVector(net.nodes_in_layer[layers-2]);
	vector *node_derivatives_this_layer = cudaBuildVector(net.nodes_in_layer[layers-3]);
	calculate_next_layer_node_derivatves<<<threadsPerBlock, blocks>>>(net, layers - 1,  *(node_outputs[layers-1]), *node_derivatives_next_layer, *node_derivatives_this_layer);



	failed |= cudaDeviceSynchronize();

	return failed;
}
int testCalculateNodeDerivatives(){
	printf("testing calculateNodeDerivatives()\n\n");
	int failed = 0;
	vector *temp = buildVector(10);

	vector **node_derivatives = calculate_node_derivatives(d_net, node_outputs, *d_expected);

	for(int layer = 0; layer < layers; layer++){
		copyDeviceToHost(node_derivatives[layer], temp);
		printVector(*temp);
	}

	return failed;
}




int testBackpropogation(){
	printf("testing backpropogation\n");
	//initializes global variables for testing
	for(int layer = 0; layer < layers; layer++){nodes[layer] = 10;}
	net = buildNetwork(layers, nodes);
	d_net = cudaBuildNetwork(layers, nodes);
	input = buildVector(nodes[0]);
	expected = buildVector(nodes[layers - 1]);
	d_input = cudaBuildVector(nodes[0]);
	d_expected = cudaBuildVector(nodes[layers - 1]);

	randomizeNetwork(net, max_weights, max_biases);
	randomizeVector(input, max_biases);
	randomizeVector(expected, max_biases);

	copyHostToDevice(&net, &d_net);
	copyHostToDevice(input, d_input);
	copyHostToDevice(expected, d_expected);

	int failed = 0;//testBackpropogate();
	failed |= testTrain();
	failed |= testCalculateNodes();
	failed |= testCalculateNextLayerWeightChanges();
	failed |= testCalculateNextLayerBiasChanges();
	failed |= testCalculateNextLayerNodeDerivatives();
	failed |= testCalculateNodeDerivatives();
	return failed;
}
