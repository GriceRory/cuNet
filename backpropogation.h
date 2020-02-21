#include "network.h"

void train(network *n, database *db, float learning_factor);//returns current cudaStatus
int backpropogate(network *d_net, network *d_change, vector *h_input, vector *d_expected);//returns current cudaStatus
vector** calculate_nodes(network *d_net, vector *d_input);

//training functions
void apply_deltas(network d_net, network d_change);//returns current cudaStatus
__global__ void calculate_next_layer_weight_changes(matrix d_change, vector d_node_outputs_next_layer, vector d_node_outputs_previous_layer, vector d_node_derivatives_next_layer);
__global__ void calculate_next_layer_bias_changes(vector d_change, vector d_node_outputs, vector d_node_derivatives);
__global__ void calculate_this_layer_node_derivatves(matrix device_connecting_weights, vector device_node_outputs_next_layer, vector device_node_derivatives_next_layer, vector device_node_derivatives_this_layer);
void calculate_last_layer_node_derivatives(vector *d_last_layer_node_derivatives, vector *d_expected_output, vector *d_node_outputs_last_layer);
vector** calculate_node_derivatives(network d_net, vector **d_node_outputs, vector *d_expected_output);//returns current cudaStatus
float correct(network d_net, database h_db, vector** possible_outputs, int number_of_possible_outputs);
vector* classify(vector v, vector **possible_outputs, int number_of_possible_outputs);
