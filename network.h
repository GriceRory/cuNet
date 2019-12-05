
#include <cuda.h>
#include <cuda_runtime.h>
#include "linear_algebra.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct network{
	int number_of_layers;
	int *nodes_in_layer;
	vector **biases;
	matrix **weights;
	void *signal_function;
};

//UTIL
int train(network *n, database db);//returns current cudaStatus
int backpropogate(network *n, float *input, float *expected);//returns current cudaStatus
int calculateNodes(network *n, float *input, float *node_outputs);
void buildNetwork(network *n, int layers, int *nodes_in_layer);

void runNetwork(network n, vector input, vector *output);
void calculateLayer(matrix weights, vector biases, vector inputs, vector *output);

//signal functions and derivative calculators
__device__ float sigmoid(float input);
__device__ float sigmoid_derivative(float output);

//training functions
int calculate_next_delta(network n, network dn, float *node_outputs);//returns current cudaStatus
int apply_deltas(network *n, network dn);//returns current cudaStatus
__global__ void calculate_next_layer_weight_changes(network n, network dn, int layer, float *node_outputs);
__global__ void calculate_next_layer_bias_changes(network dn, int layer, float *node_outputs);
__global__ void calculate_next_layer_node_derivatves(network n, int layer, float *node_outputs, float *node_derivatives);
int calculate_node_derivatives(network n, float * node_outputs, float *node_derivative);//returns current cudaStatus

//TO-DO
int train(network *n, database *sample){

}

int calculate_next_delta(network n, network dn, float *node_outputs){}
int apply_deltas(network *n, network dn){}
int calculate_next_layer_weight_changes(network n, network dn, int layer, float *node_outputs){}
int calculate_next_layer_bias_changes(network dn, int layer, float *node_outputs){}
int calculate_next_layer_node_derivatves(network n, int layer, float *node_outputs, float *node_derivatives){
	__shared__ float node_derivatives[BLOCK_SIZE];
	node_derivatives[threadIdx.x] = getElement(*(n.weights[layer]), threadIdx.x, blockIdx.x*blockDim.x)*
			signal_derivative(node_outputs_next_layer[threadIdx.x])*
			node_derivatives[threadIdx.x];
	//need to reduced sum over this then add it to the answer.
	for(int i = 2; i < BLOCK_SIZE / 2; i *= 2){
		__syncthreads();
		if(threadIdx.x < BLOCK_SIZE / i){
			node_derivatives[threadIdx.x] += node_derivatives[2*threadIdx.x];
		}
		__syncthreads();
	}
	node_output_this_layer[blockIdx.x*blockDim.x] += node_derivatives[0];
}


//COMPLETE
int calculate_node_derivatives(network n, float *node_outputs, float *node_derivative){
	int node_location = 0;
	for(int layer = 0; layer < n.number_of_layers - 1; layer++){
		node_location += n.nodes_in_layer[layer];
	}
	for(int layer = n.number_of_layers - 1; layer >= 0; layer--){
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

void buildNetwork(network *n, int layers, int *nodes_in_layer, void *function){
	n->signal_function = function;
	n->number_of_layers = layers;
	n->nodes_in_layer = malloc(sizeof(vector*)*layers);
	n->biases = malloc(layers*sizeof(vector*));
	n->weights = malloc((layers-1)*sizeof(vector*));
	for(int i = 0; i < layers - 1; i ++){
		n->nodes_in_layer[i] = nodes_in_layer[i];
		*(n->biases[i]) = buildVector(nodes_in_layer[i]);
		*(n->weights[i]) = buildMatrix(nodes_in_layer[i], nodes_in_layer[i+1]);
	}
	n->nodes_in_layer[layers-1] = nodes_in_layer[layers-1];
	*(n->biases[layers-1]) = buildvector(1, nodes_in_layer[layers-1]);
}

//given a network, input on device memory and a pointer to an output on host memory,
//calculates the output of the network on the given input.
void runNetwork(network n, vector input, vector *output){
	vector current_node_values;
	cudaBuildVector(current_node_values, 1, n.nodes_in_layer[0]);
	cudaCopyVectorHostToDevice(current_node_values, input);

	vector next_node_values;
	cudaBuildVector(next_node_values, 1, n.nodes_in_layer[1]);

	for(int current_layer = 0; current_layer < n.number_of_layers - 1; current_layer++){
		calculateLayer(n.weights[current_layer], n.biases[current_layer], current_node_values, next_node_values, n.signal_function);
	}

	cudaCopyMatrixDeviceToHost(next_node_values, *output);
}

//given the weights and biases on one layer of a network, as well as a signal function,
//calculates the next layer
int calculateLayer(matrix weights, vector biases, vector inputs, vector output, void *signal){
	int threads_per_block = BLOCK_SIZE;
	int number_of_blocks = inputs.length/BLOCK_SIZE + 1;
	matrixMultiply<<<threads_per_block, number_of_blocks>>>(inputs, weights, output);
	return_cuda_status
	number_of_blocks = output.length/BLOCK_SIZE + 1;
	matrixAdd<<<threads_per_block, number_of_blocks>>>(output, biases);
	return_cuda_status
	apply_signal_function<<<threads_per_block, number_of_blocks>>>(output, signal);
	return_cuda_status
	return cudaSuccess;
}

//signal functions and derivative calculators
__device__ float sigmoid(float input){return 1/(1+exp(-input));};
__device__ float sigmoid_derivative(float output){return output*(1-output);}
__global__ void apply_signal_function(vector v, void *signal_function){
	int idx = threadIdx.x + blockIdx.x*bloxkDim.x;
	if(idx < v.length){
		float value = (*signal_function)(getElement(m, idx));
		setElement(v, idx, value);
	}
}
