#include "linear_algebra.h"

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
void copy_database(database* source, database* target, cudaMemcpyKind copy);

void randomize_database(database h_db, float max_input, float max_output, int input_length, int output_length);
void read_vector(vector *h_v, int vector_length, FILE *file_pointer);
void write_vector(vector *h_v, FILE *file_pointer);
void free_database(database *h_db);
void cuda_free_database(database *d_db);
database* sample_database(database *db, int size);

