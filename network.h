
typedef struct{
	int number_of_layers;
	int *nodes_in_layer;
	vector **biases;
	matrix **weights;
	float (*signal_function)(float);
	float (*signal_derivative)(float);
}network;

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>




//UTIL
network buildNetwork(int layers, int *nodes_in_layer, float (*function)(float), float (*derivative)(float));
void setNetwork(network n, float max_weight, float max_bias);
int runNetwork(network n, vector input, vector *output);
int calculateLayer(matrix weights, vector biases, vector inputs, vector output, float (*signal)(float));
__device__ float getWeight(network n, int layer, int node_from, int node_to);
__device__ void setWeight(network n, int layer, int node_from, int node_to, float value);
__device__ float getBias(network n, int layer, int node);
__device__ void setBias(network n, int layer, int node, float value);

//signal functions and derivative calculators
__device__ float sigmoid(float input);
__device__ float sigmoid_derivative(float output);
__global__ void apply_signal_function(vector v, float (*signal_function)(float));

#include "backpropogation.h"

network buildNetwork(int layers, int *nodes_in_layer, float (*function)(float), float (*derivative)(float)){
	network n;
	vector v;
	matrix m;
	if(function == NULL){
		n.signal_function = &sigmoid;
		n.signal_derivative = &sigmoid_derivative;
	}else{
		n.signal_function = function;
		n.signal_derivative = derivative;
	}
	n.number_of_layers = layers;
	n.nodes_in_layer = (int *) malloc(sizeof(int)*layers);
	n.biases = (vector**)malloc(layers*sizeof(vector*));
	n.weights = (matrix**)malloc((layers-1)*sizeof(matrix*));
	for(int i = 0; i < layers - 1; i ++){
		n.nodes_in_layer[i] = nodes_in_layer[i];
		v = buildVector(nodes_in_layer[i]);
		n.biases[i] = &v;
		m = buildMatrix(nodes_in_layer[i], nodes_in_layer[i+1]);
		n.weights[i] = &m;
	}
	n.nodes_in_layer[layers-1] = nodes_in_layer[layers-1];
	v = buildVector(nodes_in_layer[layers-1]);
	n.biases[layers-1] = &v;
	return n;
}

void setNetwork(network n, float max_weight, float max_bias){
	for(int layer = 0; layer < n.number_of_layers - 1; layer++){
		randomizeVector(*(n.biases[layer]), max_bias);
		randomizeMatrix(n.weights[layer], max_weight);
	}
	randomizeVector(*(n.biases[n.number_of_layers - 1]), max_bias);
}

//given a network, input on device memory and a pointer to an output on host memory,
//calculates the output of the network on the given input.
int runNetwork(network n, vector input, vector *output){
	vector current_node_values,next_node_values;
	cudaBuildVector(&current_node_values, n.nodes_in_layer[0]);
	copyHostToDevice(&input, &current_node_values);
	cudaBuildVector(&next_node_values, n.nodes_in_layer[1]);
	for(int current_layer = 0; current_layer < n.number_of_layers - 1; current_layer++){
		calculateLayer(*n.weights[current_layer], *n.biases[current_layer], current_node_values, next_node_values, n.signal_function);
		cudaDeviceSynchronize();
	}
	copyDeviceToHost(&next_node_values, output);
	return cudaGetLastError();
}

//given the weights and biases on one layer of a network, as well as a signal function,
//calculates the next layer
int calculateLayer(matrix weights, vector biases, vector inputs, vector output, float (*signal)(float)){
	int threads_per_block = BLOCK_SIZE;
	int number_of_blocks = output.length;
	matrixMultiply<<<threads_per_block, number_of_blocks>>>(inputs, weights, output);
	cudaDeviceSynchronize();
	number_of_blocks = (output.length/BLOCK_SIZE) + 1;
	vectorAdd<<<threads_per_block, number_of_blocks>>>(output, biases);
	cudaDeviceSynchronize();
	return_cuda_status
	apply_signal_function<<<threads_per_block, number_of_blocks>>>(output, signal);
	cudaDeviceSynchronize();
	return cudaGetLastError();
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
__device__ float sigmoid(float input){return 1/(1+exp(-input));}
__device__ float sigmoid_derivative(float output){return output*(1-output);}
__global__ void apply_signal_function(vector v, float (*signal_function)(float)){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < v.length){
		float value = signal_function(getElement(v, idx));
		setElement(v, idx, value);
	}
}
