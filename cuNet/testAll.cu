int difference_tollerance(float actual, float expected, float tollerance){
	return actual-expected > tollerance || expected-actual > tollerance;
}

//25 nov 2020
//there is some device function that calls another device function in another file
//see https://stackoverflow.com/questions/45258052/visual-studio-msb3721-error-when-compiling-a-device-function-call-from-anoth
//project now compiles in windows even if it still shits itself
//there appears to be some issue with GIT that stops me from commiting that I dont want to deal with right now


//line 232 of linear_algebra.cu is having issues for some reason, commented it out.



#include <time.h>
#include <stdint.h>
cudaStream_t *streams;

#include "backpropogation.h"
#include "globals.h"



#include "testing_linear_algebra.h"
#include "testing_network.h"
#include "test_database.h"
#include "testing_backpropogation.h"
#include "test_minst.h"

int main(void){
	srand(time(NULL));
	int fails = 0;
	number_of_streams = 5;
	streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*number_of_streams);
	for(int i = 0; i < number_of_streams; ++i){
		cudaStreamCreate(&streams[i]);
	}
	
	fails = test_linear_algebra();
	if(!fails){//returns a failure failure of 1, success of 0
		printf("testing linear_algebra.h nominal\n\n\n");
	}

	int network_fails = test_network();
	fails |= network_fails;
	if(!network_fails){
		printf("testing network.h nominal\n\n\n");
	}

	int database_fails = testing_database();
	fails |= database_fails;
	if(!database_fails){printf("testing database.h nominal\n\n\n");}


	int backpropogation_fails = test_backpropogation();
	fails |= backpropogation_fails;
	if(!backpropogation_fails){printf("testing backpropogation.h nominal\n\n\n");}
	
	
	int minst_fails = test_minst();
	fails |= minst_fails;
	if(!minst_fails){printf("test_minst.h nominal\n");}
	

	if(!fails){
		printf("\e[1;1H\e[2J");
		printf("\nall systems nominal\n\n");
	}else{
		printf("\n\nsystems failure\n\n");
	}
	return fails;
}
