#include "testing_linear_algebra.h"
#include "testing_network.h"
#include "testing_backpropogation.h"
#include "test_database.h"

int main(void){
	srand(time(NULL));
	int fails = 0;


	fails = test_linear_algebra();
	if(!fails){//returns a failure failure of 1, success of 0
		printf("testing linear_algebra.h nominal\n");
	}
	int network_fails = test_network();
	fails |= network_fails;
	if(!network_fails){
		printf("testing network.h nominal\n");
	}

	int database_fails = testing_database();
	fails |= database_fails;
	if(!database_fails){printf("testing database.h nominal\n");}


	int backpropogation_fails = test_backpropogation();
	fails |= backpropogation_fails;
	if(!backpropogation_fails){printf("testing backpropogation.h nominal\n");}

	if(!fails){
		printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nall systems nominal\n\n");
	}else{printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nsystems failure\n\n");}
	return fails;
}
