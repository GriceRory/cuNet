


int test_minst();
database* build_minst_training_database();
database* build_minst_testing_database();
void initialize_minst_testing();
void test_training(int *epoc, float *probability_correct, float *best_network_probability, float *best_network_error);


void adjust_factors(float probability_correct, float error){
	if(error < 0.3){
		sample_size = 100;
		learning_factor = 0.0005;
		if(error < 0.2){
			sample_size = 200;
			learning_factor = 0.00025;
			if(error < 0.15){
				sample_size = 500;
				learning_factor = 0.0001;
				if(error < 0.125){
					sample_size = 1000;
					learning_factor = 0.00005;
					if(error < 0.1){
						sample_size = 2000;
						learning_factor = 0.000025;
						if(error < 0.075){
							sample_size = 10000;
							learning_factor = 0.000001;
							if(error < 0.05){
								sample_size = training->size;
								learning_factor = 0.00000005;
								if(error < 0.025){
									learning_factor = 0.00000001;
								}
							}
						}
					}
				}
			}
		}
	}
}

void test_training(int *epoc, float *probability_correct, float *best_network_probability, float *best_network_error){
	if(!(*epoc%epocs_per_test) && *epoc != 0){
		*probability_correct = correct(d_net, *training, possible, 10, streams, number_of_streams);

		float error = 0.0;
		for(int i = 0; i < training->size; ++i){
			error += error_term(d_net, *training->inputs[i], *training->outputs[i], streams[i%number_of_streams]);
		}
		error /= training->size;

		if(*probability_correct > *best_network_probability && *best_network_error > error){
			printf("writing network to file\n");
			copy_device_to_host(&d_net, &h_net);
			write_network(h_net, network_file_name);
		}else{
			printf("probability correct = %f, best probability = %f\nerror = %f, best error = %f\n", *probability_correct, *best_network_probability, error, *best_network_error);
		}

		if(error < *best_network_error){
			*best_network_error = error;
		}
		if(probability_correct > best_network_probability){
			best_network_probability = probability_correct;
		}
		printf("%i th epoc completed with success probability of %f, and error of %f over the training database\n", *epoc, *probability_correct, error);

		if(*probability_correct > 0.99){printf("trained to >99%% on training data");*epoc = max_epocs;}
	}else{
		database *h_training_sample = build_database(d_training_sample->size);
		copy_device_to_host(d_training_sample, h_training_sample);
		*probability_correct = correct(d_net, *h_training_sample, possible, 10, streams, number_of_streams);
		float error = 0.0;
		for(int i = 0; i < h_training_sample->size; ++i){
			error += error_term(d_net, *h_training_sample->inputs[i], *h_training_sample->outputs[i], streams[i%number_of_streams]);
		}
		error /= d_training_sample->size;
		free_database(h_training_sample);
		printf("%i th epoc completed with approximate success probability of %f, and error of %f with size %i\n", *epoc, *probability_correct, error, d_training_sample->size);
		copy_device_to_host(&d_net, &h_net);
	}
	printf("\n");
}


int test_minst(){
	initialize_minst_testing();

	float probability_correct = 0.0;//correct(d_net, *training, possible, 10, streams, number_of_streams);
	float best_network_probability = 0.1;
	float best_network_error = 4.0;
	printf("\n\nstarting training with probability %f\n\n\n", probability_correct);
	for(int epoc = 0; epoc < max_epocs; ++epoc){
		printf("%i th epoc beginning\n", epoc);

		train(&d_net, d_training_sample, learning_factor, streams, number_of_streams);
		d_training_sample = sample_database(d_training, sample_size);
		//train_with_momentum(&d_net, d_training_sample, learning_factor, streams, number_of_streams, previous_weight_and_biases, number_of_previous_weight_and_biases, 0.8);
		//printf("done training with momentum\n");
		test_training(&epoc, &probability_correct, &best_network_probability, &best_network_error);
		adjust_factors(probability_correct, best_network_error);
	}
	float testing_success_probability = correct(d_net, *testing, possible, 10, streams, number_of_streams);
	if(0.75 > testing_success_probability){
		failed = 1;
	}else{
		copy_device_to_host(&d_net, &h_net);
		write_network(h_net, "/home/rory/Documents/MNIST/training_success.bin");
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


void initialize_minst_testing(){
	printf("testing MINST\n");
	failed = 0;
	layers = 4;
	sample_size = 50;
	max_epocs = 10000;
	learning_factor = 0.001;
	max_weight = 2.0;
	max_bias = 0.5;
	epocs_per_test = 500;

	training = build_minst_training_database();
	testing = build_minst_testing_database();

	d_training = build_database(training->size);
	d_testing = build_database(testing->size);

	copy_host_to_device(training, d_training);
	copy_host_to_device(testing, d_testing);


	d_training_sample = sample_database(d_training, sample_size);

	printf("databases built\n");

	int nodes[layers];
	for(int i = 0; i < layers; ++i){
		nodes[i] = training->inputs[0]->length - ((float)(training->inputs[0]->length - training->outputs[0]->length)/(layers-1))*i;
	}

	//printf("reading network\n");
	//h_net = read_network(network_file_name);
	h_net = build_network(layers, nodes);
	randomize_network(h_net, max_weight, max_bias);
	d_net = cuda_build_network(h_net.number_of_layers, h_net.nodes_in_layer);
	copy_host_to_device(&h_net, &d_net);


	for(int i = 0; i < 10; ++i){
		possible[i] = build_vector(10);
		set_element(*possible[i], i, 1);
	}

	number_of_previous_weight_and_biases = 5;
	previous_weight_and_biases = (network**)malloc(sizeof(network*)*number_of_previous_weight_and_biases);
	for(int i = 0; i < number_of_previous_weight_and_biases; ++i){
		network net = cuda_build_network(h_net.number_of_layers, h_net.nodes_in_layer);
		previous_weight_and_biases[i] = &net;
	}
}
