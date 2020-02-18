#include <stdio.h>
#include <stdlib.h>


typedef struct{
  vector **inputs;
  vector **outputs;
  int size;
}database;


//memory management
database* build_database(int size);
int read_database(database *h_db, char *inputs, char *outputs);
int save_database(database *h_db, char *inputs, char *outputs);
int read_database_inputs(database *h_db, char *file_name);
int read_database_outputs(database *h_db, char *file_name);
int save_database_inputs(database *h_db, char *file_name);
int save_database_outputs(database *h_db, char *file_name);
void copy_host_to_device(database *host, database *device);
void copy_device_to_host(database *device, database *host);
void randomize_database(database h_db, float max_input, float max_output, int input_length, int output_length);
void read_vector(vector *h_v, int vector_length, FILE *file_pointer);
void write_vector(vector *h_v, FILE *file_pointer);
void free_database(database *h_db);
void cuda_free_database(database *d_db);
database* sample_database(database *db, int size);

database* build_database(int size){
	database *db = (database*)malloc(sizeof(database));
	db->size = size;
	db->inputs = (vector**)malloc(size*sizeof(vector*));
	db->outputs = (vector**)malloc(size*sizeof(vector*));
	return db;
}

database* sample_database(database *db, int size){
	database* sample = build_database(size);
	int *indices = (int*)malloc(sizeof(int)*size);
	for(int element = 0; element < size; ++element){
		int index = rand()%db->size;
		for(int i = 0; i < element; ++i){
			if(index == indices[i]){
				index = rand()%db->size;
				i = 0;
			}
		}
		indices[element] = index;

		sample->inputs[element] = db->inputs[indices[element]];
		sample->outputs[element] = db->outputs[indices[element]];
	}
	return sample;
}

void randomize_database(database h_db, float max_input, float max_output, int input_length, int output_length){
	for(int pair = 0; pair < h_db.size; ++pair){
		h_db.inputs[pair] = build_vector(input_length);
		h_db.outputs[pair] = build_vector(output_length);

		randomize_vector(h_db.inputs[pair], max_input);
		randomize_vector(h_db.outputs[pair], max_output);
	}
}

void read_vector(vector *h_v, int vector_length, FILE *file_pointer){
	h_v->length = vector_length;
	h_v->elements = (float *) malloc(sizeof(float)*vector_length);
	for(int element = 0; element < vector_length; element++){
		int tempLength = 40;
		char ch = fgetc(file_pointer);
		char *temp = (char *)malloc(sizeof(char)*tempLength);
		int i;
		for(i = 0; i < tempLength || ch != ',' || ch != '\n'; i++){
			temp[i] = ch;
			ch = fgetc(file_pointer);
		}
		temp[i] = '\0';
		free(temp);
		h_v->elements[element] = atof(temp);
	}
}
void write_vector(vector *h_v, FILE *file_pointer){
	for(int element = 0; element < h_v->length; element++){
		fprintf(file_pointer, "%f,",h_v->elements[element]);
	}
	fprintf(file_pointer, "\n");
}

void copy_host_to_device(database *host, database *device){
	device->size = host->size;
	for(int pair = 0; pair < host->size; pair++){
		device->inputs[pair] = cuda_build_vector(host->inputs[pair]->length);
		device->outputs[pair] = cuda_build_vector(host->outputs[pair]->length);
		copy_host_to_device(host->inputs[pair], device->inputs[pair]);
		copy_host_to_device(host->outputs[pair], device->outputs[pair]);
	}
}
void copy_device_to_host(database *device, database *host){
	host->size = device->size;
	//copying these pointers is utterly meaningless
	for(int pair = 0; pair < host->size; pair++){
		host->inputs[pair] = build_vector(device->inputs[pair]->length);
		host->outputs[pair] = build_vector(device->outputs[pair]->length);
		copy_device_to_host(device->inputs[pair], host->inputs[pair]);
		copy_device_to_host(device->outputs[pair], host->outputs[pair]);
	}
}

void free_database(database *h_db){
	for(int element = 0; element < h_db->size; element++){
		free_vector(h_db->inputs[element]);
		free_vector(h_db->outputs[element]);
	}
	free(h_db);
}

void cuda_free_database(database *d_db){
	for(int element = 0; element < d_db->size; element++){
		cuda_free_vector(d_db->inputs[element]);
		cuda_free_vector(d_db->outputs[element]);
	}
	free(d_db);
}



int read_database(database *h_db, char *inputs, char *outputs){
	int in = read_database_inputs(h_db, inputs);
	int out = read_database_outputs(h_db, outputs);
	return in || out;
}
int save_database(database *h_db, char *inputs, char *outputs){
	int in = save_database_inputs(h_db, inputs);
	int out = save_database_outputs(h_db, outputs);
	return in || out;
}


int read_database_inputs(database *h_db, char *file_name){
	FILE *inputs = fopen(file_name, "r");
	return 1;
}
int read_database_outputs(database *h_db, char *file_name){
	FILE *outputs = fopen(file_name, "r");
	return 1;
}
int save_database_inputs(database *h_db, char *file_name){
	FILE *inputs = fopen(file_name, "w");
	return 1;
}
int save_database_outputs(database *h_db, char *file_name){
	FILE *outputs = fopen(file_name, "w");
	return 1;
}
