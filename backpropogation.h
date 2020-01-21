#include "database.h"


void train(network *n, database db);//returns current cudaStatus
int backpropogate(network *d_net, network *d_change, vector *h_input, vector *d_expected);//returns current cudaStatus
vector** calculate_nodes(network *d_net, vector *h_input);

//training functions
void apply_deltas(network d_net, network d_change);//returns current cudaStatus
__global__ void calculate_next_layer_weight_changes(network d_change, int layer, vector d_node_outputs, vector d_node_derivatives);
__global__ void calculate_next_layer_bias_changes(network d_change, int layer, vector d_node_outputs, vector d_node_derivatives);
__global__ void calculate_this_layer_node_derivatves(matrix device_connecting_weights, vector device_node_outputs_next_layer, vector device_node_derivatives_next_layer, vector device_node_derivatives_this_layer);
void calculate_last_layer_node_derivatives(vector *d_last_layer_node_derivatives, vector *d_expected_output, vector *d_node_outputs_last_layer);
vector** calculate_node_derivatives(network d_net, vector **d_node_outputs, vector *d_expected_output);//returns current cudaStatus


void train(network *d_net, database *sample){
	network weight_and_bias_changes = build_network(d_net->number_of_layers, d_net->nodes_in_layer);
	for(int i = 0; i < sample->size; i++){
		network weight_and_bias_changes_sample = build_network(d_net->number_of_layers, d_net->nodes_in_layer);
		backpropogate(d_net, &weight_and_bias_changes_sample, sample->inputs[i], sample->outputs[i]);
		apply_deltas(weight_and_bias_changes, weight_and_bias_changes_sample);
	}
	apply_deltas(*d_net, weight_and_bias_changes);
}

void apply_deltas(network d_net, network d_change){
	int threadsPerBlock = 0;
	for(int layer = 0; layer < d_net.number_of_layers; layer++){
		threadsPerBlock = (d_net.weights[layer])->height;
		int blocks = (d_net.weights[layer])->width;
		matrix_add<<<threadsPerBlock, blocks>>>(*(d_net.weights[layer]), *(d_change.weights[layer]));

		threadsPerBlock = BLOCK_SIZE;
		vector_add<<<threadsPerBlock, blocks>>>(*(d_net.biases[layer]), *(d_change.biases[layer]));
	}
}

__global__ void calculate_next_layer_weight_changes(network d_change, int layer, vector d_node_outputs, vector d_node_derivatives){
	//weight from(height)
	int i = blockDim.x* blockIdx.x + threadIdx.x;
	//weight to(width)
	int j = blockDim.y* blockIdx.y + threadIdx.y;
	int nodes_in_layer = d_change.nodes_in_layer[layer];
	float dE_by_dNodeOutputNextLayer = get_element(d_node_derivatives, j + nodes_in_layer);
	float dNodeOutputNextLayer_by_dNoteInputNextLayer = sigmoid(get_element(d_node_outputs, j + nodes_in_layer));
	float dNodeInputNextLayer_by_dWeightConnecting = get_element(d_node_outputs, i);
	//the chain rule allows us to multiply these together.
	float weight_change = dE_by_dNodeOutputNextLayer * dNodeOutputNextLayer_by_dNoteInputNextLayer * dNodeInputNextLayer_by_dWeightConnecting;
	set_element(*d_change.weights[layer], i, j, weight_change);
}

__global__ void calculate_next_layer_bias_changes(network d_change, int layer, vector d_node_outputs, vector d_node_derivatives){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float dE_by_bNodeOutputNextLayer = get_element(d_node_derivatives, idx);
	float dNodeOutputNextLayer_by_dNodeInputNextLayer = sigmoid(get_element(d_node_outputs, idx));
	//dNodeInputNextLayer/dBias = 1, and using the chain rule here;
	float biasDelta = dE_by_bNodeOutputNextLayer * dNodeOutputNextLayer_by_dNodeInputNextLayer;
	set_element(*(d_change.biases[layer]), idx, biasDelta);
}

__global__ void calculate_this_layer_node_derivatves(matrix device_connecting_weights, vector device_node_outputs_next_layer, vector device_node_derivatives_next_layer, vector device_node_derivatives_this_layer){
	int nodeTo = threadIdx.x;
	int nodeFrom = blockIdx.x;
	if(nodeFrom >= device_node_derivatives_this_layer.length){return;}
	__shared__ float node_derivative_components[BLOCK_SIZE];

	for(int thread_group = 0; thread_group < (device_node_derivatives_next_layer.length/BLOCK_SIZE) + 1; thread_group++){
		nodeTo = threadIdx.x + thread_group*BLOCK_SIZE;
		if(nodeTo >= device_node_derivatives_next_layer.length){return;}

		float dE_by_dNodeOutputNextLayer = get_element(device_node_derivatives_next_layer, nodeTo);
		float dNodeOutputNextLayer_by_dNodeInputNextLayer = sigmoid_derivative(get_element(device_node_outputs_next_layer, nodeTo));
		float dNodeInputNextLayer_by_dNodeOutputThisLayerComponent = get_element(device_connecting_weights, nodeFrom, nodeTo);
		//the chain rule lets us calculate each component
		node_derivative_components[nodeTo] += dE_by_dNodeOutputNextLayer *
				dNodeOutputNextLayer_by_dNodeInputNextLayer *
				dNodeInputNextLayer_by_dNodeOutputThisLayerComponent;
	}
	reduce(node_derivative_components);

	set_element(device_node_derivatives_this_layer, nodeFrom, node_derivative_components[0]);
}

void calculate_last_layer_node_derivatives(vector *d_last_layer_node_derivatives, vector *d_expected_output, vector *d_node_outputs_last_layer){
	int threadsPerBlock = BLOCK_SIZE;
	int blocks = (d_last_layer_node_derivatives->length / BLOCK_SIZE)+1;

	vector_subtract<<<threadsPerBlock, blocks>>>(*d_last_layer_node_derivatives, *d_expected_output, *d_node_outputs_last_layer);
	scalar_multiply<<<threadsPerBlock, blocks>>>(*d_last_layer_node_derivatives, 2);
}

vector** calculate_node_derivatives(network d_net, vector **d_node_outputs, vector *d_expected_output){
	vector **d_node_derivatives = (vector**)malloc(sizeof(vector*)*d_net.number_of_layers);
	//calculation for the node derivatives in the last layer is simple.
	for(int layer = 0; layer < d_net.number_of_layers; ++layer){
		d_node_derivatives[layer] = cuda_build_vector(d_net.nodes_in_layer[layer]);
	}
	int threadsPerBlock = BLOCK_SIZE;
	int blocks;

	calculate_last_layer_node_derivatives(d_node_derivatives[d_net.number_of_layers-1], d_expected_output, d_node_outputs[d_net.number_of_layers-1]);


	//calculates each layer then checks for cuda errors.
	for(int layer = d_net.number_of_layers - 2; layer >= 0; --layer){
		blocks = d_net.nodes_in_layer[layer];
		calculate_this_layer_node_derivatves<<<threadsPerBlock, blocks>>>(*d_net.weights[layer], *d_node_outputs[layer+1], *d_node_derivatives[layer + 1], *d_node_derivatives[layer]);
		cudaDeviceSynchronize();
		sleep(2);
	}
	return d_node_derivatives;
}

int backpropogate(network *d_net, network *d_change, vector *h_input, vector *d_expected){
	int cuda_status = cudaSuccess;
	vector **node_outputs = calculate_nodes(d_net, h_input);
	vector **node_derivatives = calculate_node_derivatives(*d_net, node_outputs, d_expected);
	cuda_status = cudaGetLastError();
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

vector** calculate_nodes(network *d_net, vector *h_input){
	vector** node_outputs = (vector**)malloc(sizeof(vector*)*d_net->number_of_layers);
	node_outputs[0] = cuda_build_vector(h_input->length);
	copy_host_to_device(h_input, node_outputs[0]);
	for(int layer = 0; layer < d_net->number_of_layers - 1; layer++){
		node_outputs[layer + 1] = cuda_build_vector(d_net->nodes_in_layer[layer+1]);
		calculate_layer(*(d_net->weights[layer]), *(d_net->biases[layer]), *node_outputs[layer], *node_outputs[layer + 1]);
	}
	return node_outputs;
}
