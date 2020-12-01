//#include "database.h"

int testing_database();
int test_build_database();
int test_randomize_database();
int test_copy();
int test_read();
int test_write();


int test_build_database(){
	int size = 50;
	int failed = 0;
	database *db = build_database(size);
	failed = (size != db->size);
	if(failed){printf("failed building database\n");}
	return failed;
}
int test_randomize_database(){
	int size = 50;
	int failed = 0;
	database *db = build_database(size);
	failed = (size != db->size);
	float maxInput = 40.0;
	float maxOutput = 30.0;
	int inputLength = 10;
	int outputLength = 10;
	randomize_database(*db, maxInput, maxOutput, inputLength, outputLength);
	for(int pair = 0; pair < db->size; ++pair){
		for(int element = 0; element < db->inputs[pair]->length; ++element){
			failed = (db->inputs[pair]->elements[element] > maxInput) || (db->inputs[pair]->elements[element] < -maxInput);
			failed = (db->inputs[pair]->length != inputLength);
		}

		for(int element = 0; element < db->outputs[pair]->length; ++element){
			failed = (db->outputs[pair]->elements[element] > maxOutput) || (db->outputs[pair]->elements[element] < -maxOutput);
			failed = (db->outputs[pair]->length != outputLength);
		}
	}
	if(failed){printf("failed randomizing database\n");}
	return failed;
}
int test_copy(){
	int size = 50;
	int failed = 0;
	database *db = build_database(size);
	database *db_device = build_database(size);
	database *db_copy = build_database(size);
	failed = (size != db->size);
	float maxInput = 40.0;
	float maxOutput = 30.0;
	int inputLength = 10;
	int outputLength = 10;
	randomize_database(*db, maxInput, maxOutput, inputLength, outputLength);
	copy_host_to_device(db, db_device);
	copy_device_to_host(db_device, db_copy);
	for(int pair = 0; pair < db->size; ++pair){
		for(int element = 0; element < db->inputs[pair]->length; ++element){
			failed = (db->inputs[pair]->elements[element] > maxInput) || (db->inputs[pair]->elements[element] < -maxInput);
			failed = (db->inputs[pair]->length != inputLength);
			failed = get_element(*(db->inputs[pair]), element) != get_element(*(db_copy->inputs[pair]), element);
		}

		for(int element = 0; element < db->outputs[pair]->length; ++element){
			failed = (db->outputs[pair]->elements[element] > maxOutput) || (db->outputs[pair]->elements[element] < -maxOutput);
			failed = (db->outputs[pair]->length != outputLength);
			failed = get_element(*(db->outputs[pair]), element) != get_element(*(db_copy->outputs[pair]), element);
		}
	}
	if(failed){printf("failed randomizing database\n");}
	return failed;
}
int test_read(){
	int failed = 0;
	return failed;
}
int test_write(){
	int failed = 0;
	return failed;
}

int testing_database(){
	printf("testing database\n");
	int failed = test_build_database();
	failed |= test_randomize_database();
	failed |= test_copy();
	failed |= test_read();
	failed |= test_write();
	if(failed){printf("failed testing database\n");}
	return failed;
}
