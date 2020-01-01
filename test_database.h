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
	failed = size == db->size;
	//randomizeDatabase(*db, 40.0, 30.0, 10, 10);
	return failed;
}
int testRandomizeDatabase(){
	int size = 50;
	int failed = 0;
	database *db = buildDatabase(size);
	failed = size == db->size;
	randomizeDatabase(*db, 40.0, 30.0, 10, 10);
	return failed;
}
int testCopy(){return 0;}
int testRead(){return 0;}
int testWrite(){return 0;}

int testingDatabase(){
	printf("testing database\n");
	int failed = testBuildDatabase();
	failed |= testCopy();
	failed |= testRead();
	failed |= testWrite();
	return failed;
}
