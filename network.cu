
#include "matrix.cu"
#include <math.h>

struct network{
	int number_of_layers;
	int *nodes_in_layer;
	struct matrix *biases;
	struct matrix *weights;
};

//TO-DO
void train(network *n, database db);
__global__ void backpropogate(network *n, sample *s);
void buildNetwork(network *n, int layers, int *nodes_in_layer);

void runNetwork(network n, struct matrix input, struct matrix *output){
	struct matrix current_node_values;
	cudaBuildMatrix(current_node_values, 1, n.nodes_in_layer[0], n.nodes_in_layer[0]);
	cudaCopyMatrixHostToDevice(current_node_values, input);
	struct matrix next_node_values;
	cudaBuildMatrix(next_node_values, 1, n.nodes_in_layer[1], n.nodes_in_layer[1]);

	for(int current_layer = 0; current_layer < n.number_of_layers - 1; current_layer++){
		calculateLayer(n.weights[current_layer], n.biases[current_layer], current_node_values, next_node_values);
	}
	cudaCopyMatrixDeviceToHost(next_node_values, *output);
}

void calculateLayer(struct matrix weights, struct matrix biases, struct matrix inputs, struct matrix output){
	int threads_per_block;
	int number_of_blocks;
	matrixMultiply<<<threads_per_block, number_of_blocks>>>(inputs, weights, output);
	matrixAdd<<<threads_per_block, number_of_blocks>>>(output, biases);
	apply_signal_function<<<threads_per_block, number_of_blocks>>>(output, &sigmoid);
}


//COMPLETE



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
