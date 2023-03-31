/*
 * globals.h
 *
 *  Created on: 24/02/2020
 *      Author: rory
 */


int layers = 20;
int letters = 26;

//used in minst
vector **possible = (vector**)malloc(sizeof(vector*)*letters);
int failed, sample_size, max_epocs, epocs_per_test, number_of_streams;
float learning_factor, max_weight, max_bias;
database *training, *testing;

database *d_training, *d_testing, *d_training_sample;
char *network_file_name = (char *)"C:\\MNIST\\training.bin";//(int)(10000*probability_correct);

