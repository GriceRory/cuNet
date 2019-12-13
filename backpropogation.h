
void train(network *n, database db);//returns current cudaStatus
int backpropogate(network *n, float *input, float *expected);//returns current cudaStatus
int calculateNodes(network *n, float *input, float *node_outputs);

//training functions
int calculate_next_delta(network n, network dn, float *node_outputs);//returns current cudaStatus
void apply_deltas(network *n, network dn);//returns current cudaStatus
__global__ void calculate_next_layer_weight_changes(network dn, int layer, float *node_outputs, float *node_derivatives);
__global__ void calculate_next_layer_bias_changes(network dn, int layer, float *node_outputs, float *node_derivatives);
__global__ void calculate_next_layer_node_derivatves(network n, int layer, float *node_outputs, float *node_derivatives);
int calculate_node_derivatives(network n, float * node_outputs, float *node_derivative);//returns current cudaStatus


//TO-DO
void train(network *n, database *sample){
	for(int i = 0; i < sample->size; i++){
		backpropogate(n, sample->inputs[i], sample->outputs[i]);
	}
}

void apply_deltas(network *n, network dn){
	for(int layer = 0; layer < n->number_of_layers-2; layer++){
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 blocks(n->weights[layer]->height / BLOCK_SIZE + 1, n->weights[layer]->width / BLOCK_SIZE + 1);
		matrixAdd<<<threadsPerBlock, blocks>>>(n->weights[layer] , dn.weights[layer]);

		int threads = BLOCK_SIZE;
		int block = n->biases[layer]->length/BLOCK_SIZE + 1;
		vectorAdd<<<threads, block>>>(n->biases[layer], dn.biases[layer]);
	}
	int threads = BLOCK_SIZE;
	int block = n->biases[n->number_of_layers]->length/BLOCK_SIZE + 1;
	vectorAdd<<<threads, block>>>(n->biases[n->number_of_layers - 1], dn.biases[n->number_of_layers - 1]);
}

__global__ void calculate_next_layer_weight_changes(network dn, int layer, float *node_outputs, float *node_derivatives){
	int i = blockDim.x* blockIdx.x + threadIdx.x;
	int j = blockDim.y* blockIdx.y + threadIdx.y;
	int nodes_in_layer = dn.nodes_in_layer[layer];
	float dE_by_dNodeOutputNextLayer = node_derivatives[j + nodes_in_layer];
	float dNodeOutputNextLayer_by_dNoteInputNextLayer = (dn.signal_derivative)(node_outputs[j + nodes_in_layer]);
	float dNodeInputNextLayer_by_dWeightConnecting = node_outputs[i];
	//the chain rule allows us to multiply these together.
	float weight_change = dE_by_dNodeOutputNextLayer * dNodeOutputNextLayer_by_dNoteInputNextLayer * dNodeInputNextLayer_by_dWeightConnecting;
	setWeight(dn, layer, i, j, weight_change);
}

__global__ void calculate_next_layer_bias_changes(network n, int layer, float *node_outputs, float *node_derivatives){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float dE_by_bNodeOutputNextLayer = node_derivatives[idx + n.nodes_in_layer[layer]];
	float dNodeOutputNextLayer_by_dNodeInputNextLayer = (n.signal_derivative)(node_outputs[idx + n.nodes_in_layer[layer]]);
	//dNodeInputNextLayer/dBias = 1, and using the chain rule here;
	float biasDelta = dE_by_bNodeOutputNextLayer * dNodeOutputNextLayer_by_dNodeInputNextLayer;
	setElement(dn.biases[layer], idx, biasDelta);
}

__global__ void calculate_next_layer_node_derivatves(network n, int layer, float *node_outputs, float *node_derivatives){
	__shared__ float node_derivative_components[BLOCK_SIZE];
	float dE_by_dNodeOutputNextLayer = node_derivatives[threadIdx.x + n.nodes_in_layer[layer]];
	float dNodeOutputNextLayer_by_dNodeInputNextLayer = (n.signal_derivative)(node_outputs[threadIdx.x + n.nodes_in_layer[layer]]);
	float dNodeInputNextLayer_by_dNodeOutputThisLayerComponent = getElement(*(n.weights[layer]), threadIdx.x, blockIdx.x*blockDim.x);
	//the chain rule lets us calculate each component
	node_derivative_components[threadIdx.x] = dE_by_dNodeOutputNextLayer *
			dNodeOutputNextLayer_by_dNodeInputNextLayer *
			dNodeInputNextLayer_by_dNodeOutputThisLayerComponent;
	//and then calculate the derivative by computing their sum
	for(int i = 2; i < BLOCK_SIZE; i *= 2){
		__syncthreads();
		if(threadIdx.x < BLOCK_SIZE / i){
			node_derivative_components[threadIdx.x] += node_derivative_components[threadIdx.x + BLOCK_SIZE/i];
		}
		__syncthreads();
	}
	node_derivatives[blockIdx.x*blockDim.x] += node_derivative_components[0];
}

int calculate_node_derivatives(network n, float *node_outputs, float *node_derivative, float *expected_output){
	int node_location = 0;
	for(int layer = 0; layer < n.number_of_layers - 2; layer++){
		node_location += n.nodes_in_layer[layer];
	}

	//calculation for the node derivatives in the last layer is simple.
	for(int node = 0; node < n.nodes_in_layer[n.number_of_layers-1]; node++){//this is probably faster on CPU than transferring to a GPU
		node_derivative[node_location + node] = 2*(node_outputs[node_location + node] - expected_output[node]);
	}

	//calculates each layer then checks for cuda errors.
	for(int layer = n.number_of_layers - 2; layer >= 0; layer--){
		calculate_next_layer_node_derivatves<<<n.nodes_in_layer[layer+1], n.nodes_in_layer[layer]>>>(n, &node_outputs[node_location], &node_outputs[node_location]);

		if(cudaPeekAtLastError() != cudaSuccess){return cudaGetLastError();}
		node_location -= n.nodes_in_layer[layer];
	}
	return cudaSuccess;
}

int backpropogate(network *n, float *input, float *expected){
	int cuda_status = 0;
	float *node_outputs;
	float *node_derivatives;
	cuda_status = calculateNodes(n, input, node_outputs);
	if(cuda_status != cudaSuccess){return cuda_status;}
	cuda_status = calculate_node_derivatives(node_outputs, expected);
	if(cuda_status != cudaSuccess){return cuda_status;}
	network dn;
	buildNetwork(&dn, n->layers, n->nodes_in_layer, n->signal_function);
	for(int layer = n->number_of_layers - 1; layer <= 0; --layer){
		cuda_status = calculate_next_layer_Changes(dn, node_outputs, node_derivatives, layer);
		if(cuda_status != cudaSuccess){return cuda_status;}
	}
	return apply_deltas(n, dn);
}

int calculateNodes(network *n, float *input, float *node_outputs){
	int  cuda_status = 0;
	int number_of_nodes = 0;
	for(int layer = 0; layer < n->number_of_layers; layer++){
		number_of_nodes += n->nodes_in_layer[layer];
	}
	cudaMalloc(&node_output, number_of_nodes*sizeof(float));
	return_cuda_status
	int layer=0;
	for(float node = 0; node < number_of_nodes; node += n->nodes_in_layer[layer++]){
		vector current_node_values;
		current_node_values.length = nodes_in_layer[layer];
		current_node_values.elements = &node_outputs[node];

		vector next_node_values;
		next_node_values.length = nodes_in_layer[layer+1];
		next_node_values.elements = &node_outputs[node + nodes_in_layer[layer+1]];

		for(int current_layer = 0; current_layer < n.number_of_layers - 1; current_layer++){
			cuda_status = calculateLayer(n.weights[current_layer], n.biases[current_layer], current_node_values, next_node_values, n.signal_function);
			if(cuda_status != cudaSuccess){return cuda_status;}
		}
	}
	return cudaSuccess;
}
