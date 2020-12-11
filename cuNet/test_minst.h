


int test_minst();
database* build_minst_training_database();
database* build_minst_testing_database();
void initialize_minst_testing();



int test_minst(){
	initialize_minst_testing();
	
	printf("\n\nstarting training\n\n\n");
	float probability_correct = 0.0;//correct(d_net, *training, possible, 10);
	int epocs_with_increased_error = 0;
	for(int epoc = 0; epoc < max_epocs; ++epoc){
		printf("%i th epoc beginning\n", epoc);

		database* h_sample = training;// sample_database(training, sample_size);
		copy_database(d_training_sample, h_sample, cudaMemcpyDeviceToHost);
		float error = average_error(&d_net, h_sample, streams, number_of_streams); 
		printf("average error before = %f\t", error);

		train(&d_net, d_training, learning_factor, streams, number_of_streams);
		float error_improvement = error - average_error(&d_net, h_sample, streams, number_of_streams);
		printf("average error after = %f\t average error improvement = %f\n", error, error_improvement);
		epocs_with_increased_error += (error_improvement < 0);
		
		
		printf("epoc training complete\n");
		if(!(epoc%epocs_per_test) && epoc != 0){
			printf("calculating best learning factor\n");
			database* d_learning_factor = sample_database(d_training, sample_size);
			learning_factor = calculate_best_learning_factor(&d_net, d_learning_factor, 10, learning_factor/10, learning_factor*3, learning_factor/5, streams, number_of_streams);//0.000005;
			printf("best learning factor %f\n", learning_factor);

			printf("calculating training statistics\n");
			probability_correct = correct(d_net, *h_sample, possible, 10, streams, number_of_streams);
			printf("%i th epoc completed with success probability of %f, and error of %f\n", epoc, probability_correct, error);
			if(probability_correct > 0.975){printf("trained to >99%% on training data");break;}
			sample_size = d_training->size * exp(-3*error);
			printf("new sample size of %d", sample_size);
		}
		printf("\n\n");
	}
	copy_network(&d_net, &h_net, cudaMemcpyDeviceToHost);
	write_network(h_net, network_file_name);
	float testing_success_probability = correct(d_net, *testing, possible, 10, streams, number_of_streams);
	if(epocs_with_increased_error/max_epocs > 0.1){
		failed = 1;
		printf("failed with %f epocs_with_increased_error / max_epocs\n", epocs_with_increased_error / max_epocs);
	}

	printf("testing probability of success was %f\n", testing_success_probability);
	return failed;
}


database* build_minst_training_database(){
	int images = 60000, height = 28, width = 28;
	database *training = build_database(images);
	FILE *inputs = fopen("C:\\MNIST\\train-images.idx3-ubyte", "r");
	int32_t meta_data[4];
	fread((void *) meta_data, sizeof(int32_t), 4, inputs);

	uint8_t *image_data = (uint8_t*)malloc(sizeof(uint8_t) * height*width);
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
	FILE *outputs = fopen("C:\\MNIST\\train-labels.idx1-ubyte", "r");
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
	FILE *inputs = fopen("C:\\MNIST\\t10k-images.idx3-ubyte", "r");
	int32_t meta_data[4];
	fread((void *) meta_data, sizeof(int32_t), 4, inputs);

	uint8_t *image_data = (uint8_t*)malloc(sizeof(uint8_t) * height*width);
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
	FILE *outputs = fopen("C:\\MNIST\\t10k-labels.idx1-ubyte", "r");
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


void initialize_minst_testing(){
	printf("testing MINST\n");
	failed = 0;
	layers = 10;
	sample_size = 50;
	max_epocs = 5000;
	max_weight = 1.0;
	max_bias = 1.0;
	epocs_per_test = 20;

	training = build_minst_training_database();
	testing = build_minst_testing_database();

	d_training = build_database(training->size);
	d_testing = build_database(testing->size);

	copy_database(training, d_training, cudaMemcpyHostToDevice);
	copy_database(testing, d_testing, cudaMemcpyHostToDevice);


	d_training_sample = sample_database(d_training, sample_size);

	printf("databases built\n");

	int* nodes = (int*)malloc(layers * sizeof(int));
	for(int i = 0; i < layers; ++i){
		nodes[i] = training->inputs[0]->length - ((float)(training->inputs[0]->length - training->outputs[0]->length)/(layers-1))*i;
	}
	h_net = read_network(network_file_name);//build_network(layers, nodes);
	d_net = cuda_build_network(layers, nodes);
	randomize_network(h_net, max_weight, max_bias);
	copy_network(&h_net, &d_net, cudaMemcpyHostToDevice);
	for(int i = 0; i < 10; ++i){
		possible[i] = build_vector(10);
		set_element(*possible[i], i, 1);
	}
	
	printf("calculating best learning factor\n");
	//database *d_learning_factor = sample_database(d_training, sample_size);
	learning_factor = calculate_best_learning_factor(&d_net, d_training, 10, 0.000005, 0.00005, 0.00001, streams, number_of_streams);//0.000005;
	printf("best learning factor %f\n", learning_factor);
}
