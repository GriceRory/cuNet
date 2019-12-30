#include "database"

int testingDatabase();
int testBuildDatabase();
int testCopy();
int testRead();
int testWrite();

int testBuildDatabase(){return 0;}
int testCopy(){return 0;}
int testRead(){return 0;}
int testWrite(){return 0;}

int testingDatabase(){
	int failed = testBuildDatabase();
	failed |= testCopy();
	failed |= testRead();
	failed |= testWrite();
	return failed;
}
