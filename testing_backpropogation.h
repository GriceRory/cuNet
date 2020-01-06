

int testBackpropogation();
int testTrain();
int testBackpropogate();
int testCalculateNodes();
int testCalcualteNextDelta();
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
	vector *input = buildVector(nodes[0]);
	vector *expected = buildVector(nodes[0]);

	randomizeNetwork(n, max_weights, max_biases);
	randomizeVector(input, max_biases);

	failed |= backpropogate(&n, *input, *expected);
	printf("also here\n");

	free(nodes);
	return failed;
}
int testCalculateNodes(){
	int failed = 0;
	return failed;
}
int testCalcualteNextDelta(){
	int failed = 0;
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
	return failed;
}




int testBackpropogation(){
	printf("testing backpropogation\n");
	int failed = testBackpropogate();
	failed |= testTrain();
	failed |= testCalculateNodes();
	failed |= testCalcualteNextDelta();
	failed |= testCalculateNextLayerWeightChanges();
	failed |= testCalculateNextLayerBiasChanges();
	failed |= testCalculateNextLayerNodeDerivatives();
	failed |= testCalculateNodeDerivatives();
	return failed;
}
