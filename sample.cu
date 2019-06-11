struct sample{
  matrix input;
  matrix output;
};

#include <sample.h>

void buildSample(sample *s, int inputSize, int outputSize, int stride){
  *s.input = buildMatrix(inputSize, 1, stride);
  *s.output = buildMatrix(outputSize, 1, stride);
}
