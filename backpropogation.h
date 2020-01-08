#include "database.h"


void train(network *n, database db);//returns current cudaStatus
int backpropogate(network *d_net, network *d_change, vector h_input, vector *d_expected);//returns current cudaStatus
vector** calculate_nodes(network *d_net, vector h_input);

//training functions
void apply_deltas(network *d_net, network d_change);//returns current cudaStatus
__global__ void calculate_next_layer_weight_changes(network d_net, int layer, vector d_node_outputs, vector d_node_derivatives);
__global__ void calculate_next_layer_bias_changes(network d_net, int layer, vector d_node_outputs, vector d_node_derivatives);
__global__ void calculate_next_layer_node_derivatves(network d_net, int layer, vector d_node_outputs, vector d_node_derivatives_next_layer, vector d_node_derivatives_this_layer);
vector** calculate_node_derivatives(network d_net, vector **d_node_outputs, vector d_expected_output);//returns current cudaStatus


void train(network *d_net, database *sample){
	network weight_and_bias_changes = build_network(d_net->number_of_layers, d_net->nodes_in_layer);
	for(int i = 0; i < sample->size; i++){
		network weight_and_bias_changes_sample = build_network(d_net->number_of_layers, d_net->nodes_in_layer);
		backpropogate(d_net, &weight_and_bias_changes_sample, *(sample->inputs[i]), sample->outputs[i]);
		apply_deltas(&weight_and_bias_changes, weight_and_bias_changes_sample);
	}
	apply_deltas(d_net, weight_and_bias_changes);
}

void apply_deltas(network *d_net, network d_change){
	int threadsPerBlock = 0;
	for(int layer = 0; layer < d_net->number_of_layers; layer++){
		threadsPerBlock = (d_net->weights[layer])->height;
		int blocks = (d_net->weights[layer])->width;
		matrix_add<<<threadsPerBlock, blocks>>>(*(d_net->weights[layer]), *(d_change.weights[layer]));

		threadsPerBlock = BLOCK_SIZE;
		vector_add<<<threadsPerBlock, blocks>>>(*(d_net->biases[layer]), *(d_change.biases[layer]));
	}
}

__global__ void calculate_next_layer_weight_changes(network d_net, int layer, vector d_node_outputs, vector d_node_derivatives){
	//weight from
	int i = blockDim.x* blockIdx.x + threadIdx.x;
	//weight to
	int j = blockDim.y* blockIdx.y + threadIdx.y;
	int nodes_in_layer = d_net.nodes_in_layer[layer];
	float dE_by_dNodeOutputNextLayer = get_element(d_node_derivatives, j + nodes_in_layer);
	float dNodeOutputNextLayer_by_dNoteInputNextLayer = sigmoid(get_element(d_node_outputs, j + nodes_in_layer));
	float dNodeInputNextLayer_by_dWeightConnecting = get_element(d_node_outputs, i);
	//the chain rule allows us to multiply these together.
	float weight_change = dE_by_dNodeOutputNextLayer * dNodeOutputNextLayer_by_dNoteInputNextLayer * dNodeInputNextLayer_by_dWeightConnecting;
	set_element(*d_net.weights[layer], i, j, weight_change);
}

__global__ void calculate_next_layer_bias_changes(network d_net, int layer, vector d_node_outputs, vector d_node_derivatives){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float dE_by_bNodeOutputNextLayer = get_element(d_node_derivatives, idx);
	float dNodeOutputNextLayer_by_dNodeInputNextLayer = sigmoid(get_element(d_node_outputs, idx));
	//dNodeInputNextLayer/dBias = 1, and using the chain rule here;
	float biasDelta = dE_by_bNodeOutputNextLayer * dNodeOutputNextLayer_by_dNodeInputNextLayer;
	set_element(*(d_net.biases[layer]), idx, biasDelta);
}

__global__ void calculate_next_layer_node_derivatves(network d_net, int layer, vector d_node_outputs, vector d_node_derivatives_next_layer, vector d_node_derivatives_this_layer){
	int nodeFrom = threadIdx.x;
	int nodeTo = blockIdx.x;

	__shared__ float node_derivative_components[BLOCK_SIZE];
	float dE_by_dNodeOutputNextLayer = get_element(d_node_derivatives_next_layer, nodeFrom);
	float dNodeOutputNextLayer_by_dNodeInputNextLayer = sigmoid(get_element(d_node_outputs, nodeFrom));
	float dNodeInputNextLayer_by_dNodeOutputThisLayerComponent = 1;//get_element(*(d_net.weights[layer]), nodeFrom, nodeTo);
	//the chain rule lets us calculate each component
	node_derivative_components[nodeFrom] = dE_by_dNodeOutputNextLayer *
			dNodeOutputNextLayer_by_dNodeInputNextLayer *
			dNodeInputNextLayer_by_dNodeOutputThisLayerComponent;

	reduce(node_derivative_components);

	set_element(d_node_derivatives_this_layer, nodeTo, node_derivative_components[0]);
}

vector** calculate_node_derivatives(network d_net, vector **d_node_outputs, vector *d_expected_output){
	vector **d_node_derivatives = (vector**)malloc(sizeof(vector*)*d_net.number_of_layers);
	//calculation for the node derivatives in the last layer is simple.
	vector *h_last_layer_derivative = build_vector(d_net.nodes_in_layer[d_net.number_of_layers-1]);
	vector *d_LastLayerDerivative = cuda_build_vector(d_net.nodes_in_layer[d_net.number_of_layers-1]);
	vector *h_last_layer_outputs = build_vector(d_net.nodes_in_layer[d_net.number_of_layers-1]);
	vector *h_expected_output = build_vector(d_expected_output->length);

	copy_device_to_host(d_expected_output, h_expected_output);
	copy_device_to_host(d_node_outputs[d_net.number_of_layers - 1], h_last_layer_outputs);
	for(int node = 0; node < d_net.nodes_in_layer[d_net.number_of_layers-1]; node++){//this is faster on CPU than transferring to a GPU
		float value = 2*(get_element(*h_last_layer_outputs, node) - get_element(*h_expected_output, node));
		set_element(*h_last_layer_derivative, node, value);
	}
	copy_host_to_device(h_last_layer_derivative, d_LastLayerDerivative);
	d_node_derivatives[d_net.number_of_layers - 1] = d_LastLayerDerivative;

	//calculates each layer then checks for cuda errors.
	d_node_derivatives[d_net.number_of_layers - 1] = cuda_build_vector(d_net.nodes_in_layer[d_net.number_of_layers - 1]);
	for(int layer = d_net.number_of_layers - 2; layer >= 0; layer--){
		int threadsPerBlock = d_net.nodes_in_layer[layer];
		int blocks = d_net.nodes_in_layer[layer + 1];
		d_node_derivatives[layer] = cuda_build_vector(d_net.nodes_in_layer[layer]);
		cudaDeviceSynchronize();
		calculate_next_layer_node_derivatves<<<threadsPerBlock, blocks>>>(d_net, layer, *d_node_outputs[layer], *d_node_derivatives[layer + 1], *d_node_derivatives[layer]);
	}
	return d_node_derivatives;
}

int backpropogate(network *d_net, network *d_change, vector h_input, vector *d_expected){
	int cuda_status = cudaSuccess;
	vector **node_outputs = calculate_nodes(d_net, h_input);
	vector **node_derivatives = calculate_node_derivatives(*d_net, node_outputs, d_expected);
	if(cuda_status != cudaSuccess){return cuda_status;}
	for(int layer = d_net->number_of_layers - 2; layer >= 0; --layer){
		int threadsPerBlock = BLOCK_SIZE;
		int blocks = BLOCK_SIZE / node_outputs[layer+1]->length + 1;
		calculate_next_layer_bias_changes<<<threadsPerBlock, blocks>>>(*d_change, layer, *node_outputs[layer+1], *node_derivatives[layer+1]);
		threadsPerBlock = node_outputs[layer]->length;
		blocks = node_outputs[layer+1]->length;
		calculate_next_layer_weight_changes<<<threadsPerBlock, blocks>>>(*d_change, layer, *node_outputs[layer], *node_derivatives[layer]);
		if(cuda_status != cudaSuccess){return cuda_status;}
	}
	return cuda_status;
}

vector** calculate_nodes(network *d_net, vector h_input){
	vector** node_outputs = (vector**)malloc(sizeof(vector*)*d_net->number_of_layers);
	vector *input_device = cuda_build_vector(h_input.length);
	copy_host_to_device(&h_input, input_device);
	node_outputs[0] = input_device;
	for(int layer = 0; layer < d_net->number_of_layers - 1; layer++){
		vector *current_node_values = node_outputs[layer];
		node_outputs[layer] = current_node_values;
		vector *next_node_values = cuda_build_vector(d_net->nodes_in_layer[layer+1]);
		calculate_layer(*(d_net->weights[layer]), *(d_net->biases[layer]), *current_node_values, *next_node_values);
		node_outputs[layer + 1] = next_node_values;
	}
	return node_outputs;
}
