/*
 * globals.h
 *
 *  Created on: 24/02/2020
 *      Author: rory
 */


int layers = 20;

//used in minst
vector **possible = (vector**)malloc(sizeof(vector*)*10);
int failed, sample_size, max_epocs, epocs_per_test, number_of_streams;
float learning_factor, max_weight, max_bias;
database *training, *testing;
network **previous_weight_and_biases;
int number_of_previous_weight_and_biases;

database *d_training, *d_testing, *d_training_sample;
char *network_file_name = "/home/rory/Documents/MNIST/training.bin";
