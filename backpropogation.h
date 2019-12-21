#include "database.h"


void train(network *n, database db);//returns current cudaStatus
int backpropogate(network *n, vector input, vector expected);//returns current cudaStatus
vector** calculateNodes(network *n, vector input);

//training functions
int calculate_next_delta(network n, network dn, float *node_outputs);//returns current cudaStatus
void apply_deltas(network *n, network dn);//returns current cudaStatus
__global__ void calculate_next_layer_weight_changes(network dn, int layer, vector node_outputs, vector node_derivatives);
__global__ void calculate_next_layer_bias_changes(network dn, int layer, vector node_outputs, vector node_derivatives);
__global__ void calculate_next_layer_node_derivatves(network n, int layer, vector node_outputs, vector node_derivatives_next_layer, vector node_derivatives_this_layer);
vector** calculate_node_derivatives(network n, vector **node_outputs, vector expected_output);//returns current cudaStatus


void train(network *n, database *sample){
	for(int i = 0; i < sample->size; i++){
		backpropogate(n, *(sample->inputs[i]), *(sample->outputs[i]));
	}
}

void apply_deltas(network *n, network dn){
	int threadsPerBlock = 0;
	for(int layer = 0; layer < n->number_of_layers-2; layer++){
		threadsPerBlock = (n->weights[layer])->height;
		int blocks = (n->weights[layer])->width;
		matrixAdd<<<threadsPerBlock, blocks>>>(*(n->weights[layer]), *(dn.weights[layer]));

		threadsPerBlock = BLOCK_SIZE;
		vectorAdd<<<threadsPerBlock, blocks>>>(*(n->biases[layer]), *(dn.biases[layer]));
	}
	int block = (n->weights[n->number_of_layers - 1])->width;
	vectorAdd<<<threadsPerBlock, block>>>(*(n->biases[n->number_of_layers - 1]), *(dn.biases[n->number_of_layers - 1]));
}

__global__ void calculate_next_layer_weight_changes(network dn, int layer, vector node_outputs, vector node_derivatives){
	//weight from
	int i = blockDim.x* blockIdx.x + threadIdx.x;
	//weight to
	int j = blockDim.y* blockIdx.y + threadIdx.y;
	int nodes_in_layer = dn.nodes_in_layer[layer];
	float dE_by_dNodeOutputNextLayer = getElement(node_derivatives, j + nodes_in_layer);
	float dNodeOutputNextLayer_by_dNoteInputNextLayer = (dn.signal_derivative)(getElement(node_outputs, j + nodes_in_layer));
	float dNodeInputNextLayer_by_dWeightConnecting = getElement(node_outputs, i);
	//the chain rule allows us to multiply these together.
	float weight_change = dE_by_dNodeOutputNextLayer * dNodeOutputNextLayer_by_dNoteInputNextLayer * dNodeInputNextLayer_by_dWeightConnecting;
	setElement(*dn.weights[layer], i, j, weight_change);
}

__global__ void calculate_next_layer_bias_changes(network n, int layer, vector node_outputs, vector node_derivatives){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float dE_by_bNodeOutputNextLayer = getElement(node_derivatives, idx);
	float dNodeOutputNextLayer_by_dNodeInputNextLayer = (n.signal_derivative)(getElement(node_outputs, idx));
	//dNodeInputNextLayer/dBias = 1, and using the chain rule here;
	float biasDelta = dE_by_bNodeOutputNextLayer * dNodeOutputNextLayer_by_dNodeInputNextLayer;
	setElement(*(n.biases[layer]), idx, biasDelta);
}

__global__ void calculate_next_layer_node_derivatves(network n, int layer, vector node_outputs, vector node_derivatives_next_layer, vector node_derivatives_this_layer){
	int nodeFrom = threadIdx.x;
	int nodeTo = blockIdx.x;
	__shared__ float node_derivative_components[BLOCK_SIZE];
	float dE_by_dNodeOutputNextLayer = getElement(node_derivatives_next_layer, nodeFrom);
	float dNodeOutputNextLayer_by_dNodeInputNextLayer = (n.signal_derivative)(getElement(node_outputs, nodeFrom));
	float dNodeInputNextLayer_by_dNodeOutputThisLayerComponent = getElement(*(n.weights[layer]), nodeFrom, nodeTo);
	//the chain rule lets us calculate each component
	node_derivative_components[nodeFrom] = dE_by_dNodeOutputNextLayer *
			dNodeOutputNextLayer_by_dNodeInputNextLayer *
			dNodeInputNextLayer_by_dNodeOutputThisLayerComponent;
	//and then calculate the derivative by computing their sum
	for(int i = 2; i < BLOCK_SIZE; i *= 2){
		reduce(node_derivative_components);
	}
	setElement(node_derivatives_this_layer, nodeTo, node_derivative_components[0]);
}

vector** calculate_node_derivatives(network n, vector **node_outputs, vector expected_output){
	vector **node_derivatives;
	cudaMalloc(&node_derivatives, n.number_of_layers*sizeof(vector*));

	//calculation for the node derivatives in the last layer is simple.
	vector *LastLayerDerivative = cudaBuildVector(n.nodes_in_layer[n.number_of_layers-1]);;

	node_derivatives[n.number_of_layers - 1] = LastLayerDerivative;
	for(int node = 0; node < n.nodes_in_layer[n.number_of_layers-1]; node++){//this is probably faster on CPU than transferring to a GPU
		float value = 2*(getElement(*node_outputs[n.number_of_layers - 1], node) - getElement(expected_output, node));
		setElement(*(node_derivatives[node]), node, value);
	}

	//calculates each layer then checks for cuda errors.
	for(int layer = n.number_of_layers - 2; layer >= 0; layer--){
		vector *thisLayerDerivative = cudaBuildVector(n.nodes_in_layer[n.number_of_layers-1]);
		node_derivatives[n.number_of_layers - 1] = thisLayerDerivative;
		int threadsPerBlock = n.nodes_in_layer[layer];
		int blocks = n.nodes_in_layer[layer + 1];
		calculate_next_layer_node_derivatves<<<threadsPerBlock, blocks>>>(n, layer, *node_outputs[layer], *node_derivatives[layer + 1], *node_derivatives[layer]);
	}
	return node_derivatives;
}

int backpropogate(network *n, vector input, vector expected){
	int cuda_status = cudaSuccess;
	vector** node_outputs = calculateNodes(n, input);
	vector **node_derivatives = calculate_node_derivatives(*n, node_outputs, expected);
	if(cuda_status != cudaSuccess){return cuda_status;}
	network dn = buildNetwork(n->number_of_layers, n->nodes_in_layer, n->signal_function, n->signal_derivative);
	for(int layer = n->number_of_layers - 1; layer <= 0; --layer){
		int threadsPerBlock = BLOCK_SIZE;
		int blocks = BLOCK_SIZE / node_outputs[layer+1]->length + 1;
		calculate_next_layer_bias_changes<<<threadsPerBlock, blocks>>>(dn, layer, *node_outputs[layer+1], *node_derivatives[layer+1]);
		threadsPerBlock = node_outputs[layer]->length;
		blocks = node_outputs[layer+1]->length;;
		calculate_next_layer_weight_changes<<<threadsPerBlock, blocks>>>(dn, layer, *node_outputs[layer], *node_derivatives[layer]);
		if(cuda_status != cudaSuccess){return cuda_status;}
	}
	return cudaSuccess;
}

vector** calculateNodes(network *n, vector input){
	vector** node_outputs;

	for(int layer = 0; layer < n->number_of_layers; layer++){
		vector *current_node_values = cudaBuildVector(n->nodes_in_layer[layer]);
		cudaMemcpy(current_node_values->elements, node_outputs[layer]->elements, sizeof(float)*current_node_values->length, cudaMemcpyDeviceToDevice);

		vector *next_node_values = cudaBuildVector(n->nodes_in_layer[layer+1]);
		calculateLayer(*(n->weights[layer]), *(n->biases[layer]), *current_node_values, *next_node_values, n->signal_function);
		cudaMemcpy(node_outputs[layer + 1]->elements, next_node_values->elements, sizeof(float)*current_node_values->length, cudaMemcpyDeviceToDevice);
	}
	return node_outputs;
}
