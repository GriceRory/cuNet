
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




void free_network(network h_net);
void cuda_free_network(network d_net);
network build_network(int layers, int *nodes_in_layer);
network cuda_build_network(int layers, int *nodes_in_layer);
void randomize_network(network h_net, float max_weight, float max_bias);
int copy_host_to_device(network *host, network *device);
int copy_device_to_host(network *device, network *host);
int run_network(network d_net, vector h_input, vector *h_output);
int calculate_layer(matrix d_weights, vector d_biases, vector d_input, vector d_output);
__device__ __host__ float get_weight(network h_net, int layer, int node_from, int node_to);
__device__ __host__ void set_weight(network h_net, int layer, int node_from, int node_to, float value);
__device__ __host__ float get_bias(network h_net, int layer, int node);
__device__ __host__ void set_bias(network h_net, int layer, int node, float value);
void print_network(network h_network);
void scalar_multiply(network weight_and_bias_changes, float learning_factor);

//signal functions and derivative calculators
__device__ __host__ float sigmoid(float input);
__device__ __host__ float sigmoid_derivative(float output);
__global__ void apply_signal_function(vector v);

#include "backpropogation.h"
