
#include <cuda.h>
#include <cuda_runtime.h>
#include "linear_algebra.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "backpropogation.h"

struct network{
	int number_of_layers;
	int *nodes_in_layer;
	vector **biases;
	matrix **weights;
	void *signal_function;
	void *signal_derivative;
};


//UTIL
void buildNetwork(network *n, int layers, int *nodes_in_layer);
void runNetwork(network n, vector input, vector *output);
void calculateLayer(matrix weights, vector biases, vector inputs, vector *output);
__device__ float getWeight(network n, int layer, int node_from, int node_to);
__device__ void setWeight(network n, int layer, int node_from, int node_to, float value);
__device__ float getBias(network n, int layer, int node);
__device__ void setBias(network n, int layer, int node, float value);

//signal functions and derivative calculators
__device__ float sigmoid(float input);
__device__ float sigmoid_derivative(float output);
__global__ void apply_signal_function(vector v, void *signal_function);


void buildNetwork(network *n, int layers, int *nodes_in_layer, void *function, void *derivative){
	if(function == NULL){
		n->signal_function = &sigmoid;
		n->signal_derivative = &sigmoid_derivative;
	}
	n->signal_function = function;
	n->signal_derivative = derivative;
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

__device__ float getWeight(network n, int layer, int node_from, int node_to){
	return getElement(*(n.weights[layer]), node_from, node_to);
}
__device__ void setWeight(network n, int layer, int node_from, int node_to, float value){
	return setElement(*(n.weights[layer]), node_from, node_to, value);
}
__device__ float getBias(network n, int layer, int node){
	return getElement(*(n.biases[layer]), node);
}
__device__ void setBias(network n, int layer, int node, float value){
	return setElement(*(n.biases[layer]), node, value);
}

//signal functions and derivative calculators
__device__ float sigmoid(float input){return 1/(1+exp(-input));};
__device__ float sigmoid_derivative(float output){return output*(1-output);}
__global__ void apply_signal_function(vector v, void *signal_function){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < v.length){
		float value = (*signal_function)(getElement(m, idx));
		setElement(v, idx, value);
	}
}
