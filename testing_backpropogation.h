

int testBackpropogation();
int testTrain();
int testBackpropogate();
int testCalculateNodes();
int testCalcualteNextDelta();
int testCalculateNextLayerWeightChanges();
int testCalculateNextLayerBiasChanges();
int testCalculateNextLayerNodeDerivatives();
int testCalculateNodeDerivatives();



int testTrain(){return 0;}
int testBackpropogate(){return 0;}
int testCalculateNodes(){return 0;}
int testCalcualteNextDelta(){return 0;}
int testCalculateNextLayerWeightChanges(){return 0;}
int testCalculateNextLayerBiasChanges(){return 0;}
int testCalculateNextLayerNodeDerivatives(){return 0;}
int testCalculateNodeDerivatives(){return 0;}




int testBackpropogation(){
	int failed = testBackpropogate();
	failed |= testTrain();
	failed |= testCalculateNodes();
	failed |= testCalcualteNextDelta();
	failed |= testCalculateNextLayerWeightChanges();
	failed |= testCalculateNextLayerBiasChanges();
	failed |= testCalculateNextLayerNodeDerivatives();
	failed |= testCalculateNodeDerivatives();
	return failed;
}
