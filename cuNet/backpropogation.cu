#include "backpropogation.h"
#include <cuda.h>
#include <cuda_runtime.h>

void train(network *d_net, database *d_sample, float learning_factor, cudaStream_t *streams, int number_of_streams){
	int *nodes = (int*)malloc((d_net->number_of_layers) * sizeof(int));
	for(int i = 0; i < d_net->number_of_layers; ++i){
		nodes[i] = d_net->nodes_in_layer[i];
	}
	network weight_and_bias_changes = cuda_build_network(d_net->number_of_layers, nodes);
	for(int i = 0; i < d_sample->size; i++){
		network weight_and_bias_changes_sample = cuda_build_network(d_net->number_of_layers, d_net->nodes_in_layer);
		backpropogate(d_net, &weight_and_bias_changes_sample, d_sample->inputs[i], d_sample->outputs[i], streams[i%number_of_streams]);
		apply_deltas(weight_and_bias_changes, weight_and_bias_changes_sample, streams,  number_of_streams);
		cuda_free_network(weight_and_bias_changes_sample);
	}
	scalar_multiply(weight_and_bias_changes, learning_factor);
	apply_deltas(*d_net, weight_and_bias_changes, streams, number_of_streams);
	cuda_free_network(weight_and_bias_changes);
}

void apply_deltas(network d_net, network d_change, cudaStream_t *streams, int number_of_streams){
	int threadsPerBlock;
	int blocks;
	for(int layer = 0; layer < d_net.number_of_layers - 1; layer++){
		threadsPerBlock = (d_net.weights[layer])->height;
		blocks = (d_net.weights[layer])->width;
		matrix_add<<<blocks, threadsPerBlock, 0, streams[layer%number_of_streams]>>>(*(d_net.weights[layer]), *(d_change.weights[layer]));
		threadsPerBlock = BLOCK_SIZE;
		blocks = (d_net.biases[layer])->length/threadsPerBlock + 1;
		vector_add<<<blocks, threadsPerBlock, 0, streams[layer%number_of_streams]>>>(*(d_net.biases[layer]), *(d_change.biases[layer]));
	}
}

__global__ void calculate_next_layer_weight_changes(matrix d_change, vector d_node_outputs_next_layer, vector d_node_outputs_previous_layer, vector d_node_derivatives_next_layer){
	int row = blockDim.x* blockIdx.x + threadIdx.x;
	int col = blockDim.y* blockIdx.y + threadIdx.y;

	if(col >= d_change.width || row >= d_change.height){return;}
	float dE_by_dNodeOutputNextLayer = get_element(d_node_derivatives_next_layer, col);
	float dNodeOutputNextLayer_by_dNoteInputNextLayer = sigmoid(get_element(d_node_outputs_next_layer, col));
	float dNodeInputNextLayer_by_dWeightConnecting = get_element(d_node_outputs_previous_layer, row);
	//the chain rule allows us to multiply these together.
	float weight_change = dE_by_dNodeOutputNextLayer * dNodeOutputNextLayer_by_dNoteInputNextLayer * dNodeInputNextLayer_by_dWeightConnecting;
	set_element(d_change, row, col, weight_change);
}

__global__ void calculate_next_layer_bias_changes(vector d_change, vector d_node_outputs, vector d_node_derivatives){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= d_change.length){return;}
	float dE_by_bNodeOutputNextLayer = get_element(d_node_derivatives, idx);
	float dNodeOutputNextLayer_by_dNodeInputNextLayer = sigmoid(get_element(d_node_outputs, idx));
	//dNodeInputNextLayer/dBias = 1, and using the chain rule here;
	float biasDelta = dE_by_bNodeOutputNextLayer * dNodeOutputNextLayer_by_dNodeInputNextLayer;
	set_element(d_change, idx, biasDelta);
}

__global__ void calculate_this_layer_node_derivatves(matrix device_connecting_weights, vector device_node_outputs_next_layer, vector device_node_derivatives_next_layer, vector device_node_derivatives_this_layer){
	int nodeTo = threadIdx.x;
	int nodeFrom = blockIdx.x;
	if(nodeFrom >= device_node_derivatives_this_layer.length){return;}
	__shared__ float node_derivative_components[BLOCK_SIZE];
	node_derivative_components[threadIdx.x] = 0;
	for(; nodeTo < device_node_derivatives_next_layer.length; nodeTo += BLOCK_SIZE){
		float dE_by_dNodeOutputNextLayer = get_element(device_node_derivatives_next_layer, nodeTo);
		float dNodeOutputNextLayer_by_dNodeInputNextLayer = sigmoid_derivative(get_element(device_node_outputs_next_layer, nodeTo));
		float dNodeInputNextLayer_by_dNodeOutputThisLayerComponent = get_element(device_connecting_weights, nodeFrom, nodeTo);
		//the chain rule lets us calculate each component
		node_derivative_components[threadIdx.x] += dE_by_dNodeOutputNextLayer *
				dNodeOutputNextLayer_by_dNodeInputNextLayer *
				dNodeInputNextLayer_by_dNodeOutputThisLayerComponent;
	}
	reduce(node_derivative_components);

	set_element(device_node_derivatives_this_layer, nodeFrom, node_derivative_components[0]);
}

void calculate_last_layer_node_derivatives(vector *d_last_layer_node_derivatives, vector *d_expected_output, vector *d_node_outputs_last_layer, cudaStream_t stream){
	int threadsPerBlock = BLOCK_SIZE;
	int blocks = (d_last_layer_node_derivatives->length / BLOCK_SIZE)+1;

	vector_subtract<<<blocks, threadsPerBlock>>>(*d_last_layer_node_derivatives, *d_expected_output, *d_node_outputs_last_layer);
	scalar_multiply<<<blocks, threadsPerBlock>>>(*d_last_layer_node_derivatives, 2);
}

vector** calculate_node_derivatives(network d_net, vector **d_node_outputs, vector *d_expected_output, cudaStream_t stream){
	vector **d_node_derivatives = (vector**)malloc(sizeof(vector*)*d_net.number_of_layers);
	//calculation for the node derivatives in the last layer is simple.
	for(int layer = 0; layer < d_net.number_of_layers; ++layer){
		d_node_derivatives[layer] = cuda_build_vector(d_net.nodes_in_layer[layer]);
	}
	int threadsPerBlock = BLOCK_SIZE;
	int blocks;

	calculate_last_layer_node_derivatives(d_node_derivatives[d_net.number_of_layers-1], d_expected_output, d_node_outputs[d_net.number_of_layers-1], stream);


	//calculates each layer then checks for cuda errors.
	for(int layer = d_net.number_of_layers - 2; layer >= 0; --layer){
		blocks = d_net.nodes_in_layer[layer];
		calculate_this_layer_node_derivatves<<<blocks, threadsPerBlock>>>(*d_net.weights[layer], *d_node_outputs[layer+1], *d_node_derivatives[layer + 1], *d_node_derivatives[layer]);
		cudaStreamSynchronize(stream);
	}
	return d_node_derivatives;
}

int backpropogate(network *d_net, network *d_change, vector *d_input, vector *d_expected, cudaStream_t stream){
	int cuda_status = cudaSuccess;
	vector **node_outputs = calculate_nodes(d_net, d_input, stream);
	vector **node_derivatives = calculate_node_derivatives(*d_net, node_outputs, d_expected, stream);
	cuda_status = cudaGetLastError();
	if(cuda_status != cudaSuccess){return cuda_status;}
	for(int layer = 0; layer < d_net->number_of_layers - 1; ++layer){
		int threadsPerBlock = BLOCK_SIZE;
		int blocks = BLOCK_SIZE / node_outputs[layer+1]->length + 1;
		calculate_next_layer_bias_changes<<<blocks, threadsPerBlock, 0, stream>>>(*d_change->biases[layer], *node_outputs[layer+1], *node_derivatives[layer+1]);
		dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimBlock((d_change->weights[layer]->height/BLOCK_SIZE)+1, (d_change->weights[layer]->width/BLOCK_SIZE)+1);
		calculate_next_layer_weight_changes<<<dimGrid, dimBlock, 0, stream>>>(*d_change->weights[layer], *node_outputs[layer+1], *node_outputs[layer], *node_derivatives[layer+1]);
		if(cuda_status != cudaSuccess){return cuda_status;}
	}
	for(int i = 0; i < d_net->number_of_layers; ++i){
		cuda_free_vector(node_outputs[i]);
		cuda_free_vector(node_derivatives[i]);
	}
	free(node_outputs);
	free(node_derivatives);
	return cuda_status;
}

vector** calculate_nodes(network *d_net, vector *d_input, cudaStream_t stream){
	vector** node_outputs = (vector**)malloc(sizeof(vector*)*d_net->number_of_layers);
	node_outputs[0] = cuda_build_vector(d_input->length);
	cudaMemcpy(node_outputs[0]->elements, d_input->elements, sizeof(float)*d_input->length, cudaMemcpyDeviceToDevice);

	for(int layer = 0; layer < d_net->number_of_layers - 1; layer++){
		node_outputs[layer + 1] = cuda_build_vector(d_net->nodes_in_layer[layer+1]);
		calculate_layer(*(d_net->weights[layer]), *(d_net->biases[layer]), *node_outputs[layer], *node_outputs[layer + 1], stream);
	}
	return node_outputs;
}

//to do
//the memory management will need to be foreted out because some of these functions 
//that I am using require host memory, not device memory like I am using in thse.
float calculate_best_learning_factor(network *d_net, database *d_db, int tests_per_learning_factor, float learning_minimum, float learning_maximum, float learning_step_size, cudaStream_t *streams, int number_of_streams) {
	float best = learning_minimum;
	float best_error_improvement = 999999999999999.9;
	for (float current = learning_minimum; current < learning_maximum; current += learning_step_size) {
		float error_improvement = 0;
		for (int element = 0; element < d_db->size; ++element) {
			error_improvement += error_term(*d_net, *d_db->inputs[element], *d_db->outputs[element], streams[element % number_of_streams]);
		}

		database* temp;
		//copy_database(d_db, temp, cudaMemcpyDeviceToDevice);
		for (int i = 0; i < tests_per_learning_factor; ++i) {
			train(d_net, temp, current, streams, number_of_streams);
		}

		for (int element = 0; element < d_db->size; ++element) {
			error_improvement -= error_term(*d_net, *d_db->inputs[element], *d_db->outputs[element], streams[element % number_of_streams]);
		}

		error_improvement /= d_db->size;

		if (error_improvement < best_error_improvement) {
			best_error_improvement = error_improvement;
			best = current;
		}
	}
	return best;
}

float correct(network d_net, database h_db, vector** possible_outputs, int number_of_possible_outputs, cudaStream_t *streams, int number_of_streams){
	float probability = 0;
	vector *h_output = build_vector(1);
	for(int element = 0; element < h_db.size; ++element){
		//if(!(element%(h_db.size/10))){printf("%f of the way through measuring probability\n", ((float)element/h_db.size));}
		run_network(d_net, *h_db.inputs[element], h_output, streams[element%number_of_streams]);
		vector *classification = classify(*h_output, possible_outputs, number_of_possible_outputs);
		if(equals(*classification, *h_db.outputs[element])){
			++probability;
		}
	}
	free_vector(h_output);

	return probability/h_db.size;
}

vector* classify(vector v, vector **possible_outputs, int number_of_possible_outputs){
	float shortest = dist(v, *possible_outputs[0]);
	int index = 0;
	for(int possible = 1; possible < number_of_possible_outputs; ++possible){
		float distance = dist(v, *possible_outputs[possible]);
		if(distance < shortest){
			index = possible;
			shortest = distance;
		}
	}
	return possible_outputs[index];
}

float error_term(network d_net, vector h_input, vector h_expected, cudaStream_t stream) {
	vector* h_output = build_vector(d_net.nodes_in_layer[d_net.number_of_layers - 1]);
	run_network(d_net, h_input, h_output, stream);
	float error = dist(h_expected, *h_output);
	free_vector(h_output);
	return error;
}