#include <stdio.h>
#include<stdlib.h>

struct vector{
	int length;
	float *elements;
};

struct database{
  vector **inputs;
  vector **outputs;
  int size;
};




//memory management
int buildDatabase(database *db, char *f);
int saveDatabase(database *db, char *f);
int copyHostToDevice(database *host, database *device);
int copyDeviceToHost(database *device, database *host);

void readVector(vector *v, int vectorLength, FILE *file_pointer);
void writeVector(vector *v, File *file_pointer);

void readVector(vector *v, int vectorLength, FILE *file_pointer){
	*v = new vector;
	v->length = vectorLength;
	v->elements = malloc(sizeof(float)*vectorLength);
	for(int element = 0; element < vectorLength; element++){
		char ch = fgetc(file_pointer);
		char temp = malloc(sizeof(char)*40);
		for(int i = 0; i < tempLength || ch != ',' || ch != '\n'; i++){
			temp[i] = ch;
			ch = fgetc(file_pointer);
		}
		free(temp);
		v->elements[element] = strtof(ch, NULL);
	}
}
void writeVector(vector *v, File *file_pointer){
	for(int element = 0; element < v->length; element++){
		fprintf(file_pointer, "%f,",v->elements[element]);
	}
	fprintf(file_pointer, "\n");
}


int buildDataBase(database *db, char *f){
	FILE *file_pointer = fopen(f, "r");
	if(file_pointer == NULL){return 0;}

	db->size = readInt(file_pointer);
	int input_length = readInt(file_pointer);
	int output_length = readInt(file_pointer);
	db->inputs = malloc(sizeof(void *)*db->size);
	db->outputs = malloc(sizeof(void *)*db->size);

	for(int line = 0; line < db->size; line++){
		readVector(db->inputs[line]->elements, input_length, file_pointer);
		readVector(db->outputs[line]->elements, output_length, file_pointer);
	}
	fclose(file_pointer);
	return 1;
}
int saveDatabase(database *db, char *f){
	FILE *file_pointer = fopen(f, "w");
	if(file_pointer == NULL){return 0;}
	fprintf(file_pointer, "%d,%d,%d\n", db->size, db->inputs[0]->length, db->outputs[0]->length);
	for(int inputOutputPair = 0; inputOutputPair < db->size; inputOutputPair++){
		writeVector(db->inputs[inputOutputPair], file_pointer);
		writeVector(db->outputs[inputOutputPair], file_pointer);
	}
	fclose(file_pointer);
	return 1;
}
int copyHostToDevice(database *host, database *device){
	cudaMemcpy(device->size, host->size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&device->inputs, sizeof(void *)*host->size);
	cudaMalloc(&device->outputs, sizeof(void *)*host->size);
	for(int inputOutputPair = 0; inputOutputPair < host->size; inputOutputPair++){
		copyDeviceToHost(host->inputs[inputOutputPair], device->inputs[inputOutputPair]);
		copyDeviceToHost(host->outputs[inputOutputPair], device->outputs[inputOutputPair]);
	}
}
int copyDeviceToHost(database *device, database *host){
	cudaMemcpy(device->size, host->size, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMalloc(&host->inputs, sizeof(void *)*device->size);
	cudaMalloc(&host->outputs, sizeof(void *)*device->size);
	for(int inputOutputPair = 0; inputOutputPair < host->size; inputOutputPair++){
		copyDeviceToHost(device->inputs[inputOutputPair], host->inputs[inputOutputPair]);
		copyDeviceToHost(device->outputs[inputOutputPair], host->outputs[inputOutputPair]);
	}
}
