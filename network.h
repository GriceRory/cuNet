
typedef struct{
	int number_of_layers;
	int *nodes_in_layer;
	vector **biases;
	matrix **weights;
	//float (*signal_function)(float);
	//float (*signal_derivative)(float);
}network;

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>




//UTIL
network buildNetwork(int layers, int *nodes_in_layer);
network cudaBuildNetwork(int layers, int *nodes_in_layer);
void randomizeNetwork(network n, float max_weight, float max_bias);
int copyHostToDevice(network *host, network *device);
int copyDeviceToHost(network *device, network *host);
int runNetwork(network n, vector input, vector *output);
int calculateLayer(matrix weights, vector biases, vector inputs, vector output);
__device__ __host__ float getWeight(network n, int layer, int node_from, int node_to);
__device__ __host__ void setWeight(network n, int layer, int node_from, int node_to, float value);
__device__ __host__ float getBias(network n, int layer, int node);
__device__ __host__ void setBias(network n, int layer, int node, float value);

//signal functions and derivative calculators
__device__ __host__ float sigmoid(float input);
__device__ __host__ float sigmoid_derivative(float output);
__global__ void apply_signal_function(vector v);

#include "backpropogation.h"

network buildNetwork(int layers, int *nodes_in_layer){
	network n;
	vector *v;
	matrix *m;
	n.number_of_layers = layers;
	n.nodes_in_layer = (int *) malloc(sizeof(int)*layers);
	n.biases = (vector**)malloc(layers*sizeof(vector*));
	n.weights = (matrix**)malloc((layers-1)*sizeof(matrix*));
	for(int i = 0; i < layers - 1; i ++){
		n.nodes_in_layer[i] = nodes_in_layer[i];
		v = buildVector(nodes_in_layer[i]);
		n.biases[i] = v;
		m = buildMatrix(nodes_in_layer[i], nodes_in_layer[i+1]);
		n.weights[i] = m;
	}
	n.nodes_in_layer[layers-1] = nodes_in_layer[layers-1];
	v = buildVector(nodes_in_layer[layers-1]);
	n.biases[layers-1] = v;
	return n;
}

void setNetwork(network n, float max_weight, float max_bias){
	for(int layer = 0; layer < n.number_of_layers - 1; layer++){
		randomizeVector((n.biases[layer]), max_bias);
		randomizeMatrix(n.weights[layer], max_weight);
	}
	randomizeVector((n.biases[n.number_of_layers - 1]), max_bias);
}

network cudaBuildNetwork(int layers, int *nodes_in_layer){
	network n;
	vector *v;
	matrix *m;
	n.number_of_layers = layers;
	n.nodes_in_layer = (int *)malloc(sizeof(int)*layers);
	n.biases = (vector**)malloc(layers*sizeof(vector*));
	n.weights = (matrix**)malloc((layers-1)*sizeof(matrix*));
	for(int i = 0; i < layers - 1; i ++){
		n.nodes_in_layer[i] = nodes_in_layer[i];
		v = cudaBuildVector(nodes_in_layer[i]);
		n.biases[i] = v;
		m = cudaBuildMatrix(nodes_in_layer[i], nodes_in_layer[i+1]);
		n.weights[i] = m;
	}
	n.nodes_in_layer[layers-1] = nodes_in_layer[layers-1];
	v = cudaBuildVector(nodes_in_layer[layers-1]);
	n.biases[layers-1] = v;
	return n;
}

//given a network, input on device memory and a pointer to an output on host memory,
//calculates the output of the network on the given input.
int runNetwork(network n, vector input, vector *output){
	vector *current_node_values = cudaBuildVector(n.nodes_in_layer[0]);
	vector *next_node_values = cudaBuildVector(n.nodes_in_layer[1]);;

	copyHostToDevice(&input, current_node_values);

	for(int current_layer = 0; current_layer < n.number_of_layers - 1; current_layer++){
		calculateLayer(*n.weights[current_layer], *n.biases[current_layer], *current_node_values, *next_node_values);
		cudaDeviceSynchronize();
		sleep(2);
	}
	copyDeviceToHost(next_node_values, output);
	return cudaGetLastError();
}

//given the weights and biases on one layer of a network, as well as a signal function,
//calculates the next layer
int calculateLayer(matrix weights, vector biases, vector inputs, vector output){
	int threads_per_block = BLOCK_SIZE;
	int number_of_blocks = output.length;
	matrixMultiply<<<threads_per_block, number_of_blocks>>>(inputs, weights, output);
	cudaError_t error = cudaDeviceSynchronize();
	if(error){printf("systems failure on matrix multiply in calculateLayer calculation error: %d\n", error);return error;}

	number_of_blocks = (output.length/BLOCK_SIZE) + 1;
	vectorAdd<<<threads_per_block, number_of_blocks>>>(output, biases);
	error = cudaDeviceSynchronize();
	if(error){printf("systems failure on vector add in calculateLayer calculation error: %d\n", error);return error;}

	apply_signal_function<<<threads_per_block, number_of_blocks>>>(output);
	error = cudaDeviceSynchronize();
	if(error){printf("error type = %s\n\n", cudaGetErrorString(error));
		printf("systems failure on signal function in calculateLayer calculation error: %d\n", error);return error;}
	return error;
}

__device__ __host__ float getWeight(network n, int layer, int node_from, int node_to){
	return getElement(*(n.weights[layer]), node_from, node_to);
}
__device__ __host__ void setWeight(network n, int layer, int node_from, int node_to, float value){
	return setElement(*(n.weights[layer]), node_from, node_to, value);
}
__device__ __host__ float getBias(network n, int layer, int node){
	return getElement(*(n.biases[layer]), node);
}
__device__ __host__ void setBias(network n, int layer, int node, float value){
	return setElement(*(n.biases[layer]), node, value);
}

//signal functions and derivative calculators
__device__ __host__ float sigmoid(float input){return 1/(1+exp(-input));}
__device__ __host__ float sigmoid_derivative(float output){return output*(1-output);}
__global__ void apply_signal_function(vector v){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < v.length){
		float value = sigmoid(getElement(v, idx));//signal_function(getElement(v, idx));
		setElement(v, idx, value);
	}
}

void randomizeNetwork(network n, float max_weight, float max_bias){
	for(int layer = 0; layer < n.number_of_layers - 1; layer++){
		randomizeMatrix(n.weights[layer], max_weight);
		randomizeVector(n.biases[layer], max_bias);
	}
	randomizeVector(n.biases[n.number_of_layers - 1], max_bias);
}

int copyHostToDevice(network *host, network *device){
	device->number_of_layers = host->number_of_layers;
	int error = cudaMemcpy(device->nodes_in_layer, host->nodes_in_layer, sizeof(int)*host->number_of_layers, cudaMemcpyHostToHost);
	int temp = 0;
	if(error){printf("host to device nodes in layer error = %d\n", error);}
	for(int layer = 0; layer < host->number_of_layers - 1; layer++){
		temp = copyHostToDevice(host->weights[layer], device->weights[layer]);
		error |= temp;
		if(temp){printf("copy weights to device error %d = %d\n", layer, error);}
		temp = copyHostToDevice(host->biases[layer], device->biases[layer]);
		error |= temp;
		if(temp){printf("copy biases to device error %d = %d\n", layer, error);}
	}
	temp = copyHostToDevice(host->biases[host->number_of_layers - 1], device->biases[host->number_of_layers - 1]);
	error |= temp;
	if(temp){printf("last bias error to device %d = %d\n", host->number_of_layers - 1, error);}
	return error;
}
int copyDeviceToHost(network *device, network *host){
	host->number_of_layers = device->number_of_layers;
	int error = cudaMemcpy(host->nodes_in_layer, device->nodes_in_layer, sizeof(int)*host->number_of_layers, cudaMemcpyHostToHost);
	int temp = 0;
	if(error){printf("device to host nodes in layer error = %d\n", temp);}
	for(int layer = 0; layer < host->number_of_layers - 1; layer++){
		temp = copyDeviceToHost(device->weights[layer], host->weights[layer]);
		error |= temp;
		if(temp){printf("copy weights to host error layer = %d, error = %d\n", layer,  temp);}
		temp = copyDeviceToHost(device->biases[layer], host->biases[layer]);
		error |= temp;
		if(temp){printf("copy biases to host error layer = %d, error = %d\n", layer, temp);}
	}
	temp = copyDeviceToHost(device->biases[host->number_of_layers - 1], host->biases[host->number_of_layers - 1]);
	error |= temp;
	if(temp){printf("copy biases to host error layer = %d, error = %d\n", host->number_of_layers - 1, temp);}
	return error;
}
