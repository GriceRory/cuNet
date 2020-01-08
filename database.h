#include <stdio.h>
#include <stdlib.h>


typedef struct{
  vector **inputs;
  vector **outputs;
  int size;
}database;


//memory management
database* build_database(int size);
int read_database(database h_db, char *file_name);
int save_database(database *h_db, char *file_name);
void copy_host_to_device(database *host, database *device);
void copy_device_to_host(database *device, database *host);
void randomize_database(database h_db, float max_input, float max_output, int input_length, int output_length);
void read_vector(vector *h_v, int vector_length, FILE *file_pointer);
void write_vector(vector *h_v, FILE *file_pointer);
int read_int(FILE *file_pointer);


database* build_database(int size){
	database *db = (database*)malloc(sizeof(database));
	db->size = size;
	db->inputs = (vector**)malloc(size*sizeof(vector*));
	db->outputs = (vector**)malloc(size*sizeof(vector*));
	return db;
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

int read_int(FILE *file_pointer){
	int value = 0;
	char c = fgetc(file_pointer);
	while(c == '0' || c == '1' || c == '2' ||	c == '3' || c == '4'
			|| c == '5' || c == '6' || c == '7' || c == '8' || c == '9' ){
		value *= 10;
		value += c;
	}
	return value;
}

int readDatabase(database *db, char *f){
	int failed = 0;
	FILE *file_pointer = fopen(f, "r");
	if(file_pointer == NULL){return 1;}

	db->size = read_int(file_pointer);
	int input_length = read_int(file_pointer);
	int output_length = read_int(file_pointer);
	db->inputs = (vector**)malloc(sizeof(vector*)*db->size);
	db->outputs = (vector**)malloc(sizeof(vector*)*db->size);

	for(int line = 0; line < db->size; line++){
		read_vector(db->inputs[line], input_length, file_pointer);
		read_vector(db->outputs[line], output_length, file_pointer);
	}
	fclose(file_pointer);
	return failed;
}
int save_database(database *h_db, char *file_name){
	FILE *file_pointer = fopen(file_name, "w");
	if(file_pointer == NULL){return 0;}
	fprintf(file_pointer, "%d,%d,%d\n", h_db->size, h_db->inputs[0]->length, h_db->outputs[0]->length);
	for(int inputOutputPair = 0; inputOutputPair < h_db->size; inputOutputPair++){
		write_vector(h_db->inputs[inputOutputPair], file_pointer);
		write_vector(h_db->outputs[inputOutputPair], file_pointer);
	}
	fclose(file_pointer);
	return 1;
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
