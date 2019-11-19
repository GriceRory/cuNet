//UTIL
void train(network *n, database db);
__global__ void backpropogate(network *n, sample *s);
voi buildNetwork(network *n, int layers, int *nodes_in_layer);

void runNetwork(network n, matrix input, matrix *output);
void calculateLayer(matrix weights, matrix biases, matrix inputs, matrix *output);

//signal functions and derivative calculators
__device__ float sigmoid(float input);
__device__ float sigmoidDerivative(float output);
