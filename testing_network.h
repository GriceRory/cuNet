
#include "network.h"

int testNetwork();
int testBuildNetwork(int layers);
int testCudaBuildNetwork(int layers);
int testRandomizeNetwork(int layers);
int testCopyFunctions(int layers);
int testCalculateLayer(int layers);
int testRunNetwork(int layers);


int testBuildNetwork(int layers){
	int failed = 0;
	int nodes[layers];
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}
	network net = buildNetwork(layers, nodes, NULL, NULL);
	for(int layer = 0; layer < layers - 1; layer++){
		for(int height = 0; height < net.weights[layer]->height; height++){
			for(int width = 0; width < net.weights[layer]->width; width++){
				failed |= getWeight(net, layer, height, width) != 0.0;
				if(getWeight(net, layer, height, width) != 0.0){printf("failed weight layer = %d, height = %d, width = %d, value = %f\n\n", layer, height, width, getWeight(net, layer, height, width));}
			}
		}
		for(int element = 0; element < net.biases[layer]->length; element++){
			failed |= getBias(net, layer, element) != 0.0;
			if(getBias(net, layer, element) != 0.0){printf("failed bias layer = %d, element = %d, value =  %f\n\n", layer, element, getBias(net, layer, element));}
		}
	}
	for(int element = 0; element < net.biases[net.number_of_layers-1]->length; element++){
		failed |= getBias(net, net.number_of_layers - 1, element) != 0.0;
		if(getBias(net, net.number_of_layers - 1, element) != 0.0){printf("failed bias layer = %d, element = %d, value =  %f\n\n", net.number_of_layers, element, getBias(net, net.number_of_layers - 1, element));}
	}
	if(failed){printf("failed testing buildNetwork()");}
	return failed;
}
int testRandomizeNetwork(int layers){
	int failed = 0;
	int nodes[layers];
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}
	float weightMax = 10.0, biasMax = 20.0;
	network net = buildNetwork(layers, nodes, NULL, NULL);
	randomizeNetwork(net, weightMax, biasMax);
	for(int layer = 0; layer < layers - 1; layer++){
		for(int height = 0; height < net.weights[layer]->height; height++){
			for(int width = 0; width < net.weights[layer]->width; width++){
				failed |= getWeight(net, layer, height, width) <= -weightMax || getWeight(net, layer, height, width) >= weightMax;
				if(getWeight(net, layer, height, width) <= -weightMax || getWeight(net, layer, height, width) >= weightMax){printf("failed weight layer = %d, height = %d, width = %d, value = %f\n\n", layer, height, width, getWeight(net, layer, height, width));}
			}
		}
		for(int element = 0; element < net.biases[layer]->length; element++){
			failed |= getBias(net, layer, element) <= -biasMax || getBias(net, layer, element) >= biasMax;
			if(getBias(net, layer, element) <= -biasMax || getBias(net, layer, element) >= biasMax){printf("failed bias layer = %d, element = %d, value =  %f\n\n", layer, element, getBias(net, layer, element));}
		}
	}
	for(int element = 0; element < net.biases[net.number_of_layers-1]->length; element++){
		failed |= getBias(net, net.number_of_layers - 1, element) <= -biasMax || getBias(net, net.number_of_layers - 1, element) >= biasMax;
		if(getBias(net, net.number_of_layers - 1, element) <= -biasMax || getBias(net, net.number_of_layers - 1, element) >= biasMax){printf("failed bias layer = %d, element = %d, value =  %f\n\n", net.number_of_layers, element, getBias(net, net.number_of_layers - 1, element));}
	}
	if(failed){printf("failed testing randomizeNetwork()");}
	return failed;
}
int testCudaBuildNetwork(int layers){
	int failed = 0;
	int nodes[layers];
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}
	network net = cudaBuildNetwork(layers, nodes, NULL, NULL);
	failed |= cudaGetLastError();
	if(failed){printf("failed testing randomizeNetwork()");}
	return failed;
}
int testCopyFunctions(int layers){
	int failed = 0;
	int nodes[layers];
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}

	float weightMax = 10.0, biasMax = 20.0;
	network net = buildNetwork(layers, nodes, NULL, NULL);
	network net_copy = buildNetwork(layers, nodes, NULL, NULL);
	network net_device = cudaBuildNetwork(layers, nodes, NULL, NULL);
	randomizeNetwork(net, weightMax, biasMax);
	failed |= copyHostToDevice(&net, &net_device);
	if(failed != cudaSuccess){printf("cudaError on host to device = %d\n", failed);}
	failed = 0;
	failed |= copyDeviceToHost(&net_device, &net_copy);
	if(failed != cudaSuccess){printf("cudaError on device to host = %d\n", failed);}
	cudaDeviceSynchronize();

	/*
	for(int layer = 0; layer < layers - 1; layer++){
		if(net.nodes_in_layer[layer] != net_copy.nodes_in_layer[layer]){
			printf("failed on nodes in layer, net = %d, net_copy = %d\n", net.nodes_in_layer[layer], net_copy.nodes_in_layer[layer]);
		}
		if(net.weights[layer]->height != net_copy.weights[layer]->height || net.weights[layer]->width != net_copy.weights[layer]->width){
			printf("failed on matrix height or with net, net_copy height = %d, %d, width = %d, %d\n\n", net.weights[layer]->height, net_copy.weights[layer]->height, net.weights[layer]->width, net_copy.weights[layer]->width);
		}
		for(int row = 0; row < net.weights[layer]->height; row++){
			for(int col = 0; col < net.weights[layer]->width; col++){
				if(getWeight(net, layer, row, col) != getWeight(net_copy, layer, row, col)){
					failed = 1;
					printf("failed on weight %f != %f, layer = %d, row = %d, col = %d\n", getWeight(net, layer, row, col), getWeight(net_copy, layer, row, col), layer, row, col);
				}
			}
		}
		for(int element = 0; element < net.biases[layer]->length; element++){
			if(getBias(net, layer, element) != getBias(net_copy, layer, element)){
				failed = 1;
				printf("failed on bias %f != %f, layer = %d, element = %d\n", getBias(net, layer, element), getBias(net_copy, layer, element), layer, element);
			}
		}
	}
	for(int element = 0; element < net.biases[net.number_of_layers - 1]->length; element++){
		if(getBias(net, net.number_of_layers - 1, element) != getBias(net_copy, net.number_of_layers - 1, element)){
			failed = 1;
			printf("failed on bias %f != %f, layer = %d, element = %d\n", getBias(net, net.number_of_layers - 1, element), getBias(net_copy, net.number_of_layers - 1, element), net.number_of_layers - 1, element);
		}
	}*/
	if(failed){printf("\n\nfailed testing copyDeviceToHost() and copyHostToDevice() for networks\n");}
	return failed;
}
int testCalculateLayer(int layers){
	int failed = 0;
	int nodes[layers];
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}

	float weightMax = 10.0, biasMax = 20.0;
	network net = buildNetwork(layers, nodes, NULL, NULL);
	vector *input = buildVector(net.nodes_in_layer[0]);
	randomizeVector(input, biasMax);
	vector *output = buildVector(net.nodes_in_layer[1]);
	randomizeNetwork(net, weightMax, biasMax);
	calculateLayer(*(net.weights[0]), *(net.biases[0]), *input, *output, net.signal_function);
	for(int col = 0; col < (net.weights[0])->width; col++){
		float temp = 0.0;
		for(int row = 0; row < (net.weights[0])->height; row++){
			temp += getElement(*(net.weights[0]), row, col) * getElement(*input, row);
		}
		temp = (*(net.signal_function))(temp);
		if(getElement(*output, col) - temp > 1 || getElement(*output, col) - temp < -1){
			printf("failed on index %d with out = %.10f, expected = %.10f\n", col, getElement(*output, col), temp);
			failed = 1;
		}
	}
	if(failed){printf("failed testing calculateLayer()");}
	return failed;
}
int testRunNetwork(int layers){
	int failed = 0;
	int nodes[layers];
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}

	float weightMax = 10.0, biasMax = 20.0;
	network net = buildNetwork(layers, nodes, NULL, NULL);
	randomizeNetwork(net, weightMax, biasMax);
	network net_device = cudaBuildNetwork(layers, nodes, NULL, NULL);
	copyHostToDevice(&net, &net_device);
	vector *input = buildVector(net.nodes_in_layer[0]);
	randomizeVector(input, biasMax);
	vector *output = buildVector(net.nodes_in_layer[1]);
	runNetwork(net_device, *input, output);

	for(int layer = 0; layer < layers - 1; layer++){
		vector *nextLayer = buildVector(net.nodes_in_layer[layer+1]);
		for(int col = 0; col < (net.weights[0])->width; col++){
			float temp = 0.0;
			for(int row = 0; row < (net.weights[0])->height; row++){
				temp += getElement(*(net.weights[0]), row, col) * getElement(*input, row);
			}
			temp = (*(net.signal_function))(temp);
			setElement(*nextLayer, col, temp);
		}
		freeVector(input);
		input = nextLayer;
	}
	for(int element = 0; element < output->length; element++){
		if(!(getElement(*output, element) - getElement(*input, element) < 0.01 && getElement(*output, element) - getElement(*input, element) > -0.01)){
			printf("failed on output element = %d, output = %f, expected = %f\n", element, getElement(*output, element), getElement(*input, element));
			failed = 1;
		}
	}
	if(failed){printf("failed testing runNetwork()\n");}
	return failed;
}



int testNetwork(){
	int failed = 0;
	printf("testing network.h\n");
	failed |= testBuildNetwork(5);
	failed |= testCudaBuildNetwork(5);
	//failed |= testRandomizeNetwork(5);
	failed |= testCopyFunctions(5);
	//failed |= testCalculateLayer(5);
	//failed |= testRunNetwork(3);
	printf("\nFinished testing network.h\n\n");
	return failed;
}
