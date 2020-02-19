


int test_minst();
database* build_minst_training_database();
database* build_minst_testing_database();



int test_minst(){
	printf("testing MINST\n");
	int failed = 0;
	int sample_size = 300;
	database *training = build_minst_training_database();
	database *testing = build_minst_testing_database();

	database *d_training = build_database(training->size);
	database *d_testing = build_database(testing->size);

	copy_host_to_device(training, d_training);
	copy_host_to_device(testing, d_testing);


	database *d_training_sample = sample_database(d_training, sample_size);

	printf("databases built\n");
	int layers = 10;
	int nodes[layers];
	for(int i = 0; i < layers; ++i){
		nodes[i] = training->inputs[0]->length - ((float)(training->inputs[0]->length - training->outputs[0]->length)/(layers-1))*i;
	}
	network net = build_network(layers, nodes);
	network d_net = cuda_build_network(layers, nodes);
	randomize_network(net, 5.0, 2.0);
	copy_host_to_device(&net, &d_net);

	vector **possible = (vector**)malloc(sizeof(vector*)*10);
	for(int i = 0; i < 10; ++i){
		possible[i] = build_vector(10);
		set_element(*possible[i], i, 1);
	}

	printf("\n\nstarting training\n\n\n");
	float probability_correct = 0.0;//correct(d_net, *training, possible, 10);
	for(int epoc = 0; epoc < 10000; ++epoc){
		printf("%i th epoc beginning\n", epoc);
		train(&d_net, d_training_sample, 0.005);
		printf("epoc training complete\n");
		d_training_sample = sample_database(d_training, sample_size+=10);
		printf("sampling database complete\n");

		if(!(epoc%10)){
			printf("calculating training statistics\n");
			probability_correct = correct(d_net, *training, possible, 10);
			printf("probability calculated %f\n", probability_correct);
			float error = 0.0;
			for(int i = 0; i < training->size; ++i){
				error += error_term(d_net, *training->inputs[i], *training->outputs[i]);
			}
			printf("%i th epoc completed with success probability of %f, and error of %f\n", epoc, probability_correct, error);
			if(probability_correct >  0.99){break;}
		}
		printf("\n\n");
	}

	float testing_success_probability = correct(d_net, *testing, possible, 10);
	if(0.75 > testing_success_probability){
		failed = 1;
	}

	printf("testing probability of success was %f\n", testing_success_probability);
	return failed;
}


database* build_minst_training_database(){
	int images = 60000, height = 28, width = 28;
	database *training = build_database(images);
	FILE *inputs = fopen("/home/rory/Documents/MNIST/train-images.idx3-ubyte", "r");
	int32_t meta_data[4];
	fread((void *) meta_data, sizeof(int32_t), 4, inputs);

	uint8_t image_data[height*width];
	for(int image = 0; image < images; image++){
		fread((void *) image_data, sizeof(uint8_t), height*width, inputs);
		training->inputs[image] = build_vector(height*width);
		for(int element = 0; element < height * width; element++){
			set_element(*(training->inputs[image]), element, (float)image_data[element]);
		}
	}
	fclose(inputs);
	printf("read training inputs\n");

	uint8_t image_label;
	FILE *outputs = fopen("/home/rory/Documents/MNIST/train-labels.idx1-ubyte", "r");
	fread((void *) meta_data, sizeof(int32_t), 2, outputs);
	for(int image = 0; image < images; image++){
		fread((void*) &image_label, sizeof(uint8_t), 1, outputs);
		training->outputs[image] = build_vector(10);
		set_element(*training->outputs[image], image_label, 1);
	}
	fclose(outputs);
	printf("read training outputs\n");
	return training;
}

database* build_minst_testing_database(){
	int images = 10000, height = 28, width = 28;
	database *testing = build_database(images);
	FILE *inputs = fopen("/home/rory/Documents/MNIST/t10k-images.idx3-ubyte", "r");
	int32_t meta_data[4];
	fread((void *) meta_data, sizeof(int32_t), 4, inputs);

	uint8_t image_data[height*width];
	for(int image = 0; image < images; image++){
		fread((void *) image_data, sizeof(uint8_t), height*width, inputs);
		testing->inputs[image] = build_vector(height*width);
		for(int element = 0; element < height * width; element++){
			set_element(*(testing->inputs[image]), element, (float)image_data[element]);
		}
	}
	fclose(inputs);
	printf("read testing inputs\n");

	uint8_t image_label;
	FILE *outputs = fopen("/home/rory/Documents/MNIST/t10k-labels.idx1-ubyte", "r");
	fread((void *) meta_data, sizeof(int32_t), 2, outputs);
	for(int image = 0; image < images; image++){
		fread((void*) &image_label, sizeof(uint8_t), 1, outputs);
		testing->outputs[image] = build_vector(10);
		set_element(*testing->outputs[image], image_label, 1);
	}
	fclose(outputs);
	printf("read testing outputs\n");
	return testing;
}
