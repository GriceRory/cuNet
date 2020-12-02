#include "network.h"

network build_network(int layers, int *nodes_in_layer){
	network n;
	n.number_of_layers = layers;
	n.nodes_in_layer = (int *) malloc(sizeof(int)*layers);
	n.biases = (vector**)malloc((layers-1)*sizeof(vector*));
	n.weights = (matrix**)malloc((layers-1)*sizeof(matrix*));
	for(int i = 0; i < layers-1; ++i){
		n.nodes_in_layer[i] = nodes_in_layer[i];
		n.biases[i] = build_vector(nodes_in_layer[i+1]);
		n.weights[i] = build_matrix(nodes_in_layer[i], nodes_in_layer[i+1]);
	}
	n.nodes_in_layer[layers-1] = nodes_in_layer[layers-1];
	return n;
}

network cuda_build_network(int layers, int *nodes_in_layer){
	network n;
	n.number_of_layers = layers;
	n.nodes_in_layer = (int *)malloc(sizeof(int)*layers);
	n.biases = (vector**)malloc((layers-1)*sizeof(vector*));
	n.weights = (matrix**)malloc((layers-1)*sizeof(matrix*));
	for(int i = 0; i < layers - 1; i ++){
		n.nodes_in_layer[i] = nodes_in_layer[i];
		n.biases[i] = cuda_build_vector(nodes_in_layer[i+1]);
		n.weights[i] = cuda_build_matrix(nodes_in_layer[i], nodes_in_layer[i+1]);
	}
	n.nodes_in_layer[layers-1] = nodes_in_layer[layers-1];
	return n;
}

void scalar_multiply(network d_net, float learning_factor){
	for(int layer = 0; layer < d_net.number_of_layers - 1; layer++){
		int threads_per_block = BLOCK_SIZE;
		int blocks = (d_net.weights[layer]->height) * (d_net.weights[layer]->width)/threads_per_block +1;
		scalar_multiply<<<blocks, threads_per_block>>>(*d_net.weights[layer], learning_factor);
		blocks = d_net.biases[layer]->length/threads_per_block + 1;
		scalar_multiply<<<blocks, threads_per_block>>>(*d_net.biases[layer], learning_factor);
	}
}

//given a network, input on device memory and a pointer to an output on host memory,
//calculates the output of the network on the given input.
int run_network(network d_net, vector h_input, vector *h_output, cudaStream_t stream){
	vector *current_node_values = cuda_build_vector(d_net.nodes_in_layer[0]);
	vector *next_node_values = cuda_build_vector(d_net.nodes_in_layer[1]);
	copy_host_to_device(&h_input, current_node_values);
	for(int current_layer = 0; current_layer < d_net.number_of_layers - 1; current_layer++){
		calculate_layer(*d_net.weights[current_layer], *d_net.biases[current_layer], *current_node_values, *next_node_values, stream);
		cuda_free_vector(current_node_values);
		current_node_values = (vector*)malloc(sizeof(vector*));
		current_node_values = next_node_values;
		next_node_values = cuda_build_vector(d_net.nodes_in_layer[current_layer + 2]);
	}
	copy_device_to_host(current_node_values, h_output);
	cuda_free_vector(current_node_values);
	cuda_free_vector(next_node_values);
	return cudaGetLastError();
}

//given the weights and biases on one layer of a network, as well as a signal function,
//calculates the next layer
int calculate_layer(matrix d_weights, vector d_biases, vector d_input, vector d_output, cudaStream_t stream){
	int threads_per_block = BLOCK_SIZE;
	int number_of_blocks = d_output.length;
	matrix_multiply<<<number_of_blocks, threads_per_block, 0, stream>>>(d_input, d_weights, d_output);
	number_of_blocks = (d_output.length/BLOCK_SIZE) + 1;
	vector_add<<<number_of_blocks, threads_per_block>>>(d_output, d_biases);
	apply_signal_function<<<number_of_blocks, threads_per_block>>>(d_output);
	int error = cudaStreamSynchronize(stream);
	if(error){printf("error type = %s\n\n", cudaGetErrorString((cudaError_t) error));
	printf("systems failure on signal function in calculateLayer calculation error: %s\n", cudaGetErrorName((cudaError_t) error));return error;}
	return error;
}

__device__ __host__ float get_weight(network h_net, int layer, int node_from, int node_to){
	return get_element(*(h_net.weights[layer]), node_from, node_to);
}
__device__ __host__ void set_weight(network h_net, int layer, int node_from, int node_to, float value){
	return set_element(*(h_net.weights[layer]), node_from, node_to, value);
}
__device__ __host__ float get_bias(network h_net, int layer, int node){
	return get_element(*(h_net.biases[layer]), node);
}
__device__ __host__ void set_bias(network h_net, int layer, int node, float value){
	return set_element(*(h_net.biases[layer]), node, value);
}

//signal functions and derivative calculators
__device__ __host__ float sigmoid(float input){
	return 1/(1+exp(-input));
}
__device__ __host__ float sigmoid_derivative(float output){
	return output*(1-output);
}
__global__ void apply_signal_function(vector v){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < v.length){
		float value = sigmoid(get_element(v, idx));//signal_function(getElement(v, idx));
		set_element(v, idx, value);
	}
}

void randomize_network(network h_net, float max_weight, float max_bias){
	for(int layer = 0; layer < h_net.number_of_layers - 1; layer++){
		randomize_matrix(h_net.weights[layer], max_weight);
		randomize_vector(h_net.biases[layer], max_bias);
	}
}

int copy_host_to_device(network *host, network *device){
	device->number_of_layers = host->number_of_layers;
	printf("1\n");
	int error = cudaMemcpy(device->nodes_in_layer, host->nodes_in_layer, sizeof(int)*host->number_of_layers, cudaMemcpyHostToHost);
	printf("2\n");
	int temp = 0;
	printf("3\n");
	if(error){printf("host to device nodes in layer error = %d\n", error);}
	for(int layer = 0; layer < host->number_of_layers - 1; layer++){
		printf("layer %d of %d\n", layer, host->number_of_layers);
		temp = copy_matrix(host->weights[layer], device->weights[layer], cudaMemcpyHostToDevice);
		printf("layer %d of %d\n", layer, host->number_of_layers);
		error |= temp;
		if(temp){printf("copy weights to device error %d = %d\n", layer, error);}
		temp = copy_host_to_device(host->biases[layer], device->biases[layer]);
		error |= temp;
		if(temp){printf("copy biases to device error %d = %d\n", layer, error);}
	}
	return error;
}
int copy_device_to_host(network *device, network *host){
	host->number_of_layers = device->number_of_layers;
	int error = cudaMemcpy(host->nodes_in_layer, device->nodes_in_layer, sizeof(int)*host->number_of_layers, cudaMemcpyHostToHost);
	int temp = 0;
	if(error){printf("device to host nodes in layer error = %d\n", temp);}
	for(int layer = 0; layer < host->number_of_layers - 1; layer++){
		temp = copy_matrix(device->weights[layer], host->weights[layer], cudaMemcpyDeviceToHost);
		error |= temp;
		if(temp){printf("copy weights to host error layer = %d, error = %d\n", layer,  temp);}
		temp = copy_device_to_host(device->biases[layer], host->biases[layer]);
		error |= temp;
		if(temp){printf("copy biases to host error layer = %d, error = %d\n", layer, temp);}
	}
	return error;
}


void free_network(network h_net){
	free(h_net.nodes_in_layer);
	for(int layer = 0; layer < h_net.number_of_layers - 1; layer++){
		free_vector(h_net.biases[layer]);
		free_matrix(h_net.weights[layer]);
	}
}
void cuda_free_network(network d_net){
	free(d_net.nodes_in_layer);
	for(int layer = 0; layer < d_net.number_of_layers - 1; layer++){
		cuda_free_vector(d_net.biases[layer]);
		cuda_free_matrix(d_net.weights[layer]);
	}
}

void print_network(network h_network){
	printf("number of layers %d\n nodes in each layer: ", h_network.number_of_layers);
	for(int layer = 0; layer < h_network.number_of_layers - 1; ++layer){
		printf("%d, ", h_network.nodes_in_layer[layer]);
	}
	printf("%d\n", h_network.nodes_in_layer[h_network.number_of_layers-1]);

	for(int layer = 0; layer < h_network.number_of_layers - 1; ++layer){
		printf("layer %d weights and biases\n", layer);
		printf("\nweights\n\n");
		print_matrix(*h_network.weights[layer]);
		printf("\nbiases\n\n");
		print_vector(*h_network.biases[layer]);
	}
}

void write_network(network h_net, char *file_name){
	FILE *outputs = fopen(file_name, "w");
	fwrite(&h_net.number_of_layers, sizeof(int), 1, outputs);
	fwrite(h_net.nodes_in_layer, sizeof(int), h_net.number_of_layers, outputs);
	for(int i = 0; i < h_net.number_of_layers-1; ++i){
		fwrite(h_net.weights[i]->elements, sizeof(float), h_net.weights[i]->height * h_net.weights[i]->width, outputs);
		fwrite(h_net.biases[i]->elements, sizeof(float), h_net.biases[i]->length, outputs);
	}
	fclose(outputs);
}

network read_network(char *file_name){
	int number_of_layers = 0;
	FILE *inputs = fopen(file_name, "r");
	fread(&number_of_layers, sizeof(int), 1, inputs);
	int *nodes = (int*)malloc(sizeof(int)*number_of_layers);
	fread(nodes, sizeof(int), number_of_layers, inputs);
	network h_net = build_network(number_of_layers, nodes);
	for(int i = 0; i < h_net.number_of_layers-1; ++i){
		fread(h_net.weights[i]->elements, sizeof(float), h_net.weights[i]->height * h_net.weights[i]->width, inputs);
		fread(h_net.biases[i]->elements, sizeof(float), h_net.biases[i]->length, inputs);
	}
	fclose(inputs);
	return h_net;
}
