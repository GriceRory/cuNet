/*
 * globals.h
 *
 *  Created on: 24/02/2020
 *      Author: rory
 */


int layers = 20;
int numbers = 10;

//used in minst
vector **possible = (vector**)malloc(sizeof(vector*)*numbers);
int failed, sample_size, max_epocs, epocs_per_test, number_of_streams;
float learning_factor, max_weight, max_bias;
database *training, *testing, *training_sample;

char* training_images_IDX = (char*)"C:\\MNIST\\train-images.idx3-ubyte";
char* training_labels_IDX = (char*)"C:\\MNIST\\train-labels.idx1-ubyte";
char* testing_images_IDX = (char*)"C:\\MNIST\\t10k-images.idx3-ubyte";
char* testing_labels_IDX = (char*)"C:\\MNIST\\t10k-labels.idx1-ubyte";



database *d_training, *d_testing, *d_training_sample;
char *network_file_name = (char *)"C:\\MNIST\\training.bin";//(int)(10000*probability_correct);

