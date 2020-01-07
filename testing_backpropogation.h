int testBackpropogation();
int testTrain();
int testBackpropogate();
int testCalculateNodes();
int testCalculateNextLayerWeightChanges();
int testCalculateNextLayerBiasChanges();
int testCalculateNextLayerNodeDerivatives();
int testCalculateNodeDerivatives();



int testTrain(){
	int failed = 0;
	return failed;
}
int testBackpropogate(){
	int failed = 0;
	int layers = 5;
	float max_weights = 2.0;
	float max_biases = 1.0;

	int *nodes = (int*)malloc(sizeof(int)*layers);

	for(int layer = 0; layer < layers; layer++){nodes[layer] = 10;}

	network n = buildNetwork(layers, nodes);
	network d_n = cudaBuildNetwork(layers, nodes);
	vector *input = buildVector(nodes[0]);
	vector *expected = buildVector(nodes[0]);

	randomizeNetwork(n, max_weights, max_biases);
	randomizeVector(input, max_biases);
	copyHostToDevice(&n, &d_n);

	network weight_and_bias_changes = cudaBuildNetwork(layers, nodes);
	failed |= backpropogate(&d_n, &weight_and_bias_changes, *input, *expected);

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
	int layers = 5;
	float biasMax = 1.0;
	float weightMax = 1.0;

	int *nodes = (int*)malloc(sizeof(int)*layers);

	for(int layer = 0; layer < layers; layer++){nodes[layer] = 10;}

	network net = buildNetwork(layers, nodes);
	network d_n = cudaBuildNetwork(layers, nodes);
	vector *input = buildVector(nodes[0]);
	vector *d_input = cudaBuildVector(nodes[0]);

	randomizeVector(input, biasMax);
	randomizeNetwork(net, weightMax, biasMax);

	copyHostToDevice(&net, &d_n);
	copyHostToDevice(input, d_input);

	vector **node_outputs = calculateNodes(&d_n, *input);

	vector **node_outputs_host = (vector**)malloc(sizeof(vector*)*net.number_of_layers);
	for(int layer = 0; layer < layers; layer++){
		vector *temp = buildVector(10);
		copyDeviceToHost(node_outputs[layer], temp);
		node_outputs_host[layer] = temp;
	}


	for(int layer = 0; layer < layers - 1; layer++){
		vector *nextLayer = buildVector(net.nodes_in_layer[layer+1]);
		for(int col = 0; col < (net.weights[0])->width; col++){
			float temp = 0.0;
			for(int row = 0; row < (net.weights[0])->height; row++){
				temp += getElement(*(net.weights[0]), row, col) * getElement(*input, row);
			}
			temp = sigmoid(temp);
			setElement(*nextLayer, col, temp);
		}
		for(int element = 0; element < nextLayer->length; element++){
			if(!(getElement(*node_outputs_host[layer], element) - getElement(*input, element) < 0.9 && getElement(*node_outputs_host[layer], element) - getElement(*input, element) > -0.9)){
				printf("failed in layer %d on output element = %d, output = %f, expected = %f\n", layer, element, getElement(*node_outputs_host[layer], element), getElement(*input, element));
				failed = 1;
			}
		}
		freeVector(input);
		input = nextLayer;
	}
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
	return failed;
}
int testCalculateNodeDerivatives(){
	int failed = 0;
	int layers = 10;
	float biasMax = 1.0;
	float weightMax = 1.0;

	int *nodes = (int*)malloc(sizeof(int)*10);
	for(int layer = 0; layer < layers; layer++){
		nodes[layer] = 10;
	}

	network net = buildNetwork(layers, nodes);
	vector *input = buildVector(nodes[0]);
	vector *output = buildVector(nodes[layers-1]);

	network device_net = cudaBuildNetwork(layers, nodes);
	vector *device_input = cudaBuildVector(nodes[0]);
	vector *device_output = cudaBuildVector(nodes[layers-1]);

	randomizeNetwork(net, weightMax, biasMax);
	randomizeVector(input, biasMax);
	randomizeVector(output, biasMax);

	copyHostToDevice(&net, &device_net);
	copyHostToDevice(input, device_input);
	copyHostToDevice(output, device_output);


	vector **node_outputs = calculateNodes(&device_net, *device_input);
	vector **node_derivatives = calculate_node_derivatives(device_net, node_outputs, *device_output);

	node_derivatives = calculate_node_derivatives(device_net, node_outputs, *device_output);

	vector *temp = buildVector(10);
	for(int layer = 0; layer < layers; layer++){
		copyDeviceToHost(node_derivatives[layer], temp);
		printVector(*temp);
	}

	return failed;
}




int testBackpropogation(){
	printf("testing backpropogation\n");
	int failed = 0;//testBackpropogate();
	failed |= testTrain();
	failed |= testCalculateNodes();
	failed |= testCalculateNextLayerWeightChanges();
	failed |= testCalculateNextLayerBiasChanges();
	failed |= testCalculateNextLayerNodeDerivatives();
	failed |= testCalculateNodeDerivatives();
	return failed;
}
