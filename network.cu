
#include "matrix.cu"
#include <math.h>

struct network{
	int number_of_layers;
	int *nodes_in_layer;
	matrix **biases;
	matrix **weights;
	void *signal_function;
};

//TO-DO
void train(network *n, database *sample){}

float* calculateNodes(network *n, float *input){
	int number_of_nodes = 0;
	for(int layer = 0; layer < n->number_of_layers; layer++){
		number_of_nodes += n->nodes_in_layer[layer];
	}
	float *node_outputs;
	cudaMalloc(&node_output, number_of_nodes*sizeof(float));

	int layer=0;
	for(float node = 0; node < number_of_nodes; node += n->nodes_in_layer[layer++]){
		matrix current_node_values;
		current_node_values.height = 1;
		current_node_values.width = current_node_values.stride = nodes_in_layer[layer];
		current_node_values.elements = &node_outputs[node];

		matrix next_node_values;
		next_node_values.height = 1;
		next_node_values.width = next_node_values.stride = nodes_in_layer[layer+1];
		next_node_values.elements = &node_outputs[node + nodes_in_layer[layer+1]];

		for(int current_layer = 0; current_layer < n.number_of_layers - 1; current_layer++){
			calculateLayer(n.weights[current_layer], n.biases[current_layer], current_node_values, next_node_values, n.signal_function);
		}
	}
	return node_outputs;
}

void backpropogate(network *n, float *input, float *expected){
	float node_outputs = calculateNodes(n, input);

}


//COMPLETE

void buildNetwork(network *n, int layers, int *nodes_in_layer, void *function){
	n->signal_function = function;
	n->number_of_layers = layers;
	n->nodes_in_layer = malloc(sizeof(int)*layers);
	n->biases = malloc(layers*sizeof(struct matrix*));
	n->weights = malloc((layers-1)*sizeof(struct matrix*));
	for(int i = 0; i < layers - 1; i ++){
		n->nodes_in_layer[i] = nodes_in_layer[i];
		*(n->biases[i]) = buildMatrix(1, nodes_in_layer[i]);
		*(n->weights[i]) = buildMatrix(nodes_in_layer[i], nodes_in_layer[i+1]);
	}
	n->nodes_in_layer[layers-1] = nodes_in_layer[layers-1];
	*(n->biases[layers-1]) = buildMatrix(1, nodes_in_layer[layers-1]);
}

//given a network, input on device memory and a pointer to an output on host memory,
//calculates the output of the network on the given input.
void runNetwork(network n, matrix input, matrix *output){
	matrix current_node_values;
	cudaBuildMatrix(current_node_values, 1, n.nodes_in_layer[0]);
	cudaCopyMatrixHostToDevice(current_node_values, input);

	matrix next_node_values;
	cudaBuildMatrix(next_node_values, 1, n.nodes_in_layer[1]);

	for(int current_layer = 0; current_layer < n.number_of_layers - 1; current_layer++){
		calculateLayer(n.weights[current_layer], n.biases[current_layer], current_node_values, next_node_values, n.signal_function);
	}

	cudaCopyMatrixDeviceToHost(next_node_values, *output);
}

//given the weights and biases on one layer of a network, as well as a signal function,
//calculates the next layer
void calculateLayer(matrix weights, matrix biases, matrix inputs, matrix output, void *signal){
	int threads_per_block;
	int number_of_blocks;
	matrixMultiply<<<threads_per_block, number_of_blocks>>>(inputs, weights, output);
	matrixAdd<<<threads_per_block, number_of_blocks>>>(output, biases);
	apply_signal_function<<<threads_per_block, number_of_blocks>>>(output, signal);
}

//signal functions and derivative calculators
__device__ float sigmoid(float input){return 1/(1+exp(-input));};
__device__ float sigmoidDerivative(float output){return output*(1-output);}
__global__ void apply_signal_function(struct matrix m, void *signal_function){
	int idx = threadIdx.x + blockIdx.x*bloxkDim.x;
	if(idx < m.width){
		float value = (*signal_function)(getElement(m, 0, idx));
		setElement(m, 0, idx, value);
	}
}
