//util
__device__ void setWeightDelta(trainingData data, int layer, int firingNode, int targetNode, float weight);
__device__ float getWeightDelta(trainingData data, int layer, int firingNode, int targetNode);

__device__ void setBiasDelta(trainingData data, int layer, int node, float bias);
__device__ float getBiasDelta(trainingData data, int layer, int node);

__device__ void setNodeOutput(trainingData data, int layer, int node, float output);
__device__ float getNodeOutput(trainingData data, int leyer, int node);


//memory
void buildTrainingData(trainingData *data, struct network n, sample s);
void destroyTrainingData(trainingData *data);
