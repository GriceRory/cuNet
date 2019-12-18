#include <stdio.h>
#include <stdlib.h>

typedef struct{
  vector **inputs;
  vector **outputs;
  int size;
}database;


//memory management
int buildDatabase(database *db, char *f);
int saveDatabase(database *db, char *f);
int copyHostToDevice(database *host, database *device);
int copyDeviceToHost(database *device, database *host);

void readVector(vector *v, int vectorLength, FILE *file_pointer);
void writeVector(vector *v, FILE *file_pointer);
int readInt(FILE *file_pointer);

void readVector(vector *v, int vectorLength, FILE *file_pointer){
	v->length = vectorLength;
	v->elements = (float *) malloc(sizeof(float)*vectorLength);
	for(int element = 0; element < vectorLength; element++){
		int tempLength = 40;
		char ch = fgetc(file_pointer);
		char *temp = (char *)malloc(sizeof(char)*tempLength);
		for(int i = 0; i < tempLength || ch != ',' || ch != '\n'; i++){
			temp[i] = ch;
			ch = fgetc(file_pointer);
		}
		free(temp);
		v->elements[element] = strtof(ch, NULL);
	}
}
void writeVector(vector *v, FILE *file_pointer){
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
	device->size = host->size;
	cudaMalloc(&device->inputs, sizeof(void *)*host->size);
	cudaMalloc(&device->outputs, sizeof(void *)*host->size);
	for(int inputOutputPair = 0; inputOutputPair < host->size; inputOutputPair++){
		copyDeviceToHost(host->inputs[inputOutputPair], device->inputs[inputOutputPair]);
		copyDeviceToHost(host->outputs[inputOutputPair], device->outputs[inputOutputPair]);
	}
}
int copyDeviceToHost(database *device, database *host){
	host->size = device->size;
	cudaMalloc(&host->inputs, sizeof(void *)*device->size);
	cudaMalloc(&host->outputs, sizeof(void *)*device->size);
	for(int inputOutputPair = 0; inputOutputPair < host->size; inputOutputPair++){
		copyDeviceToHost(device->inputs[inputOutputPair], host->inputs[inputOutputPair]);
		copyDeviceToHost(device->outputs[inputOutputPair], host->outputs[inputOutputPair]);
	}
}
