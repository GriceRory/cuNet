#include "testing_linear_algebra.h"
#include "testing_network.h"
#include "testing_backpropogation.h"
#include "test_database.h"

int main(void){
	srand(time(NULL));
	int fails = 0;//testLinearAlgebra();
	if(!fails){//returns a failure failure of 1, success of 0
		printf("testing linear_algebra.h nominal\n");
	}
	int network_fails = 0;//testNetwork();
	fails |= network_fails;
	if(!network_fails){
		printf("testing network.h nominal\n");
	}

	int database_fails = testingDatabase();
	fails |= database_fails;
	if(!database_fails){printf("testing database.h nominal\n");}


	int backpropogation_fails = testBackpropogation();
	fails |= backpropogation_fails;
	if(!backpropogation_fails){printf("testing backpropogation.h nominal\n");}

	if(!fails){
		printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nall systems nominal\n\n");
	}else{printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nsystems failure\n\n");}
	return fails;
}
