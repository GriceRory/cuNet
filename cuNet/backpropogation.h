#include "network.h"

void train(network *n, database *db, float learning_factor, cudaStream_t *streams, int number_of_streams);//returns current cudaStatus
int backpropogate(network *d_net, network *d_change, vector *h_input, vector *d_expected, cudaStream_t stream);//returns current cudaStatus
vector** calculate_nodes(network *d_net, vector *d_input, cudaStream_t stream);

//training functions
void apply_deltas(network d_net, network d_change, cudaStream_t *streams, int number_of_streams);//returns current cudaStatus
__global__ void calculate_next_layer_weight_changes(matrix d_change, vector d_node_outputs_next_layer, vector d_node_outputs_previous_layer, vector d_node_derivatives_next_layer);
__global__ void calculate_next_layer_bias_changes(vector d_change, vector d_node_outputs, vector d_node_derivatives);
__global__ void calculate_this_layer_node_derivatves(matrix device_connecting_weights, vector device_node_outputs_next_layer, vector device_node_derivatives_next_layer, vector device_node_derivatives_this_layer);
void calculate_last_layer_node_derivatives(vector *d_last_layer_node_derivatives, vector *d_expected_output, vector *d_node_outputs_last_layer, cudaStream_t stream);
vector** calculate_node_derivatives(network d_net, vector **d_node_outputs, vector *d_expected_output, cudaStream_t stream);//returns current cudaStatus


//not tested yet
float probability_correct(network d_net, database h_db, vector** possible_outputs, int number_of_possible_outputs, cudaStream_t *streams, int number_of_streams);
vector* classify(vector v, vector **possible_outputs, int number_of_possible_outputs);
float error_term(network d_net, vector h_input, vector h_expected, cudaStream_t stream);
float calculate_best_learning_factor(network* d_net, database* d_db, int tests_per_learning_factor, float learning_minimum, float learning_maximum, float learning_step_size, cudaStream_t* streams, int number_of_streams);
float calculate_improvement(database* h_db, network* d_net, network* h_net, int tests_per_learning_factor, database* d_db, cudaStream_t* s, int streams, float learning_factor);	
float average_error(network* d_net, database* h_sample, cudaStream_t* streams, int number_of_streams);