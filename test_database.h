//#include "database.h"

int testingDatabase();
int testBuildDatabase();
int testRandomizeDatabase();
int testCopy();
int testRead();
int testWrite();


int testBuildDatabase(){
	int size = 50;
	int failed = 0;
	database *db = buildDatabase(size);
	failed = (size != db->size);
	if(failed){printf("failed building database\n");}
	return failed;
}
int testRandomizeDatabase(){
	int size = 50;
	int failed = 0;
	database *db = buildDatabase(size);
	failed = (size != db->size);
	float maxInput = 40.0;
	float maxOutput = 30.0;
	int inputLength = 10;
	int outputLength = 10;
	randomizeDatabase(*db, maxInput, maxOutput, inputLength, outputLength);
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
int testCopy(){
	int size = 50;
	int failed = 0;
	database *db = buildDatabase(size);
	database *db_device = buildDatabase(size);
	database *db_copy = buildDatabase(size);
	failed = (size != db->size);
	float maxInput = 40.0;
	float maxOutput = 30.0;
	int inputLength = 10;
	int outputLength = 10;
	randomizeDatabase(*db, maxInput, maxOutput, inputLength, outputLength);
	copyHostToDevice(db, db_device);
	copyDeviceToHost(db_device, db_copy);
	for(int pair = 0; pair < db->size; ++pair){
		for(int element = 0; element < db->inputs[pair]->length; ++element){
			failed = (db->inputs[pair]->elements[element] > maxInput) || (db->inputs[pair]->elements[element] < -maxInput);
			failed = (db->inputs[pair]->length != inputLength);
			failed = getElement(*(db->inputs[pair]), element) != getElement(*(db_copy->inputs[pair]), element);
		}

		for(int element = 0; element < db->outputs[pair]->length; ++element){
			failed = (db->outputs[pair]->elements[element] > maxOutput) || (db->outputs[pair]->elements[element] < -maxOutput);
			failed = (db->outputs[pair]->length != outputLength);
			failed = getElement(*(db->outputs[pair]), element) != getElement(*(db_copy->outputs[pair]), element);
		}
	}
	if(failed){printf("failed randomizing database\n");}
	return failed;
}
int testRead(){
	int failed = 0;
	return failed;
}
int testWrite(){
	int failed = 0;
	return failed;
}

int testingDatabase(){
	printf("testing database\n");
	int failed = testBuildDatabase();
	failed |= testRandomizeDatabase();
	failed |= testCopy();
	failed |= testRead();
	failed |= testWrite();
	if(failed){printf("failed testing database\n");}
	return failed;
}
