//core training
void train(network *n, database db);
__global__ void backpropogate(network *n, sample *s);

void runNetwork(network n, smaple *s);
void calculateLayer(matrix weights, matrix biases, matrix inputs, matrix *output);

__device__ float sigmoid(float input);
