
#include "network.h"

int testNetwork();
int testBuildNetwork();


int testBuildNetwork(int layers){
	int failed = 0;
	int nodes[layers];
	for(int i = 0; i < layers; i++){
		nodes[i] = 10;
	}
	network net = buildNetwork(layers, nodes, NULL, NULL);
	return failed;
}

int testNetwork(){
	int failed = 0;
	printf("testing network.h");
	testBuildNetwork(5);
	printf("\nFinished testing network.h\n\n");
	return failed;
}
