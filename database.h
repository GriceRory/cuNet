#include <stdio.h>
#include <stdlib.h>


typedef struct{
  vector **inputs;
  vector **outputs;
  int size;
}database;


//memory management
database* buildDatabase(int size);
int readDatabase(database db, char *f);
int saveDatabase(database *db, char *f);
void copyHostToDevice(database *host, database *device);
void copyDeviceToHost(database *device, database *host);
void randomizeDatabase(database db, float maxInput, float maxOutput, int inputLength, int outputLength);
void readVector(vector *v, int vectorLength, FILE *file_pointer);
void writeVector(vector *v, FILE *file_pointer);
int readInt(FILE *file_pointer);


database* buildDatabase(int size){
	database *db = (database*)malloc(sizeof(database));
	db->size = size;
	db->inputs = (vector**)malloc(size*sizeof(vector*));
	db->outputs = (vector**)malloc(size*sizeof(vector*));
	return db;
}

void randomizeDatabase(database db, float maxInput, float maxOutput, int inputLength, int outputLength){
	for(int pair = 0; pair < db.size; ++pair){
		db.inputs[pair] = buildVector(inputLength);
		db.outputs[pair] = buildVector(outputLength);

		randomizeVector(db.inputs[pair], maxInput);
		randomizeVector(db.outputs[pair], maxOutput);
	}
}

void readVector(vector *v, int vectorLength, FILE *file_pointer){
	v->length = vectorLength;
	v->elements = (float *) malloc(sizeof(float)*vectorLength);
	for(int element = 0; element < vectorLength; element++){
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
		v->elements[element] = atof(temp);
	}
}
void writeVector(vector *v, FILE *file_pointer){
	for(int element = 0; element < v->length; element++){
		fprintf(file_pointer, "%f,",v->elements[element]);
	}
	fprintf(file_pointer, "\n");
}

int readInt(FILE *file_pointer){
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

	db->size = readInt(file_pointer);
	int input_length = readInt(file_pointer);
	int output_length = readInt(file_pointer);
	db->inputs = (vector**)malloc(sizeof(vector*)*db->size);
	db->outputs = (vector**)malloc(sizeof(vector*)*db->size);

	for(int line = 0; line < db->size; line++){
		readVector(db->inputs[line], input_length, file_pointer);
		readVector(db->outputs[line], output_length, file_pointer);
	}
	fclose(file_pointer);
	return failed;
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
void copyHostToDevice(database *host, database *device){
	device->size = host->size;
	for(int pair = 0; pair < host->size; pair++){
		device->inputs[pair] = cudaBuildVector(host->inputs[pair]->length);
		device->outputs[pair] = cudaBuildVector(host->outputs[pair]->length);
		copyHostToDevice(host->inputs[pair], device->inputs[pair]);
		copyHostToDevice(host->outputs[pair], device->outputs[pair]);
	}
}
void copyDeviceToHost(database *device, database *host){
	host->size = device->size;
	//copying these pointers is utterly meaningless
	for(int pair = 0; pair < host->size; pair++){
		host->inputs[pair] = buildVector(device->inputs[pair]->length);
		host->outputs[pair] = buildVector(device->outputs[pair]->length);
		copyDeviceToHost(device->inputs[pair], host->inputs[pair]);
		copyDeviceToHost(device->outputs[pair], host->outputs[pair]);
	}
}
