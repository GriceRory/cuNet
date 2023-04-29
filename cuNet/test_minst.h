


int test_minst();
database* build_minst_training_database();
database* build_minst_testing_database();
void initialize_minst_testing();
void print_minst_letter(vector h_letter);
vector** read_IDX(char* file);
int get_magic(int8_t magic);
vector* read_vector(uint8_t* data, int size, int length, int letter);






int test_minst(){
	initialize_minst_testing();
	/*
	printf("\n\nstarting training\n\n\n");
	database* h_sample = sample_database(training, sample_size);
	print_minst_letter(*(h_sample->inputs[0]));
	for (int i = 0; i < 5; ++i) { 
		print_minst_letter(*(h_sample->inputs[i])); 
		print_vector(*(h_sample->outputs[i]));
	}
	
	float probability_correct = correct(d_net, *h_sample, possible, 10, streams, number_of_streams);
	int epocs_with_increased_error = 0;
	for(int epoc = 0; epoc < max_epocs; ++epoc){
		printf("%i th epoc beginning\n", epoc);

		h_sample = sample_database(training, sample_size);
		copy_database(d_training_sample, h_sample, cudaMemcpyDeviceToHost);
		float error = average_error(&d_net, h_sample, streams, number_of_streams); 
		printf("average error before = %f\t", error);

		train(&d_net, d_training_sample, learning_factor, streams, number_of_streams);
		float error_improvement = error - average_error(&d_net, h_sample, streams, number_of_streams);
		printf("average error after = %f\t average error improvement = %f\n", error, error_improvement);
		epocs_with_increased_error += (error_improvement < 0);
		
		
		printf("epoc training complete\n");
		if(!(epoc%epocs_per_test) && epoc != 0){
			printf("calculating best learning factor\n");
			database* d_learning_factor = sample_database(d_training, sample_size);
			learning_factor = calculate_best_learning_factor(&d_net, d_learning_factor, 5, learning_factor/8, learning_factor*2, learning_factor/3, streams, number_of_streams);//0.000005;
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
	*/
	return failed;
}

unsigned int reverse(unsigned int x)
{
	unsigned int swapped = ((x >> 24) & 0xff) | // move byte 3 to byte 0
		((x << 8) & 0xff0000) | // move byte 1 to byte 2
		((x >> 8) & 0xff00) | // move byte 2 to byte 1
		((x << 24) & 0xff000000); // byte 0 to byte 3
	return swapped;

}

database* build_minst_training_database(){
	database* training = build_database(60000);
	free(training->inputs);
	free(training->outputs);
	training->inputs = read_IDX(training_images_IDX);
	training->outputs = read_IDX(training_labels_IDX);
	return training;
}

vector** read_IDX(char* file) {
	FILE* f = fopen(file, "r");
	uint8_t magic_number[4];
	size_t bytes_read = fread((void*)magic_number, sizeof(uint8_t), 4, f);
	int size = get_magic(magic_number[2]);
	int items;
	bytes_read = fread((void *) &items, sizeof(int), 1, f);
	items = reverse(items);

	int dimentions = magic_number[3] - 1;
	int* dimention = (int*)malloc(sizeof(int)*dimentions);

	bytes_read = fread((void*)dimention, sizeof(int), dimentions, f);

	int length = 1;
	for (int i = 0; i < dimentions; ++i) {
		length *= reverse(dimention[i]);
	}
	vector** output = (vector**)malloc(items * sizeof(vector*));
	
	uint8_t* datastream = (uint8_t*)malloc(items*length*size);
	
	bytes_read = fread((void*)datastream, size, length*items, f);
	for (int item = 0; item < items; ++item) {
		output[item] = read_vector(datastream, size, length, item);
	}
	fclose(f);
	return output;
}

database* build_minst_testing_database(){
	database *testing = build_database(10000);
	free(testing->inputs);
	free(testing->outputs);
	testing->inputs = read_IDX(testing_images_IDX);
	testing->outputs = read_IDX(testing_labels_IDX);
	return testing;
}

void initialize_minst_testing(){
	printf("testing MINST\n");
	//constants
	failed = 0;
	layers = 10;
	sample_size = 200;
	max_epocs = 5000;
	max_weight = 1.0;
	max_bias = 1.0;
	epocs_per_test = 20;

	//build image databases
	training = build_minst_training_database();
	testing = build_minst_testing_database();

	d_training = build_database(training->size);
	d_testing = build_database(testing->size);

	copy_database(training, d_training, cudaMemcpyHostToDevice);
	copy_database(testing, d_testing, cudaMemcpyHostToDevice);
	
	//sample image database
	training_sample = sample_database(training, sample_size);
	d_training_sample = build_database(sample_size);
	copy_database(training_sample, d_training_sample, cudaMemcpyHostToDevice);

	printf("databases built\n");
	

	//building host and device networks
	int* nodes = (int*)malloc(layers * sizeof(int));
	
	float input_length = (float)training->inputs[0]->length;
	float output_length = (float)training->outputs[0]->length;
	
	float gradient = ((input_length - output_length) / (layers - 1));
	
	for(int i = 0; i < layers; ++i){
		nodes[i] = input_length - gradient*i;
	}

	h_net = build_network(layers, nodes);
	d_net = cuda_build_network(layers, nodes);
	randomize_network(h_net, max_weight, max_bias);
	copy_network(&h_net, &d_net, cudaMemcpyHostToDevice);

	//building the set of possible vectors
	for(int i = 0; i < numbers; ++i){
		possible[i] = build_vector(numbers);
		set_element(*possible[i], i, 1);
	}
	
	//finds the best learning factor at the start
	printf("calculating best learning factor\n");
	database* h_learning_factor = sample_database(training, sample_size);
	database* d_learning_factor = build_database(sample_size);
	copy_database(h_learning_factor, d_learning_factor, cudaMemcpyHostToDevice);
	learning_factor = calculate_best_learning_factor(&d_net, d_learning_factor, 10, 0.0, 100.0, 10.0, streams, number_of_streams);
	printf("best learning factor %f\n", learning_factor);
}

void print_minst_letter(vector h_letter) {
	for (int element = 0; element < h_letter.length; ++element) {	
		if (!(element % 28)) {
			printf("\n");
		}
		if ((int)get_element(h_letter, element) < 0X10) {
			printf("0");
		}
		printf("%x ", (int)get_element(h_letter, element));
	}
	printf("\n\n\n");
}

vector* read_vector(uint8_t* data, int size, int length, int item) {
	vector* vector = build_vector(length);
	for (int element = 0; element < length; ++element) {
		set_element(*vector, element, (float)data[item * length + element]);
	}
	return vector;
}

int get_magic(int8_t magic) {
	int size = -1;
	switch (magic) {
	case (0x08): size = sizeof(char);
		break;
	case(0x09): size = sizeof(char);
		break;
	case(0x0B): size = sizeof(short);
		break;
	case(0x0C): size = sizeof(int);
		break;
	case(0x0D): size = sizeof(float);
		break;
	case(0x0E): size = sizeof(double);
		break;
	default: printf("invalid magic number\n");
	}
	return size;
}