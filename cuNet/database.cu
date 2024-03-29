#include "database.h"

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
	free(indices);
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


void copy_database(database* source, database* target, cudaMemcpyKind copy) {
	target->size = source->size;
	for (int pair = 0; pair < target->size; pair++) {
		if (copy == cudaMemcpyHostToDevice || copy == cudaMemcpyDeviceToDevice) {
			target->inputs[pair] = cuda_build_vector(source->inputs[pair]->length);
			target->outputs[pair] = cuda_build_vector(source->outputs[pair]->length);
		}else {
			target->inputs[pair] = build_vector(source->inputs[pair]->length);
			target->outputs[pair] = build_vector(source->outputs[pair]->length);
		}
		copy_vector(source->inputs[pair], target->inputs[pair], copy);
		copy_vector(source->outputs[pair], target->outputs[pair], copy);
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
