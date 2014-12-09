#include <iostream>
using std::cout;
using std::cin;
using std::endl;

#include <vector>

#define DLL_SVM_SUPPORT
#define DLL_PARALLEL

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/ocv_visualizer.hpp"
#include "dll/test.hpp"

#include "dll/cpp_utils/data.hpp"

#include <cstdio>

#define PATCH_SIZE 21
#define NB_PATCHES 2000

typedef double T;
#define READ_FEATURE_FORMAT ",%lf"
#define DATA_FILE "parzival_test_21_2000.dat"
#define RBM_FILE "rbm_parzival_21_200000.dat"
#define DBN_SVM_FILE "dbm_svm_parzival_21_200000.dat"


void read_data(char* data_filename, std::vector<std::vector<T>>& samples, std::vector<std::size_t>& labels);
int get_nblines(char* filename);
int get_nbcolumns(char* filename);

void rbm_features();
void train_dbn_svm();
void svm_predict();
void svm_classify();
int doesFileExist(const char *filename);

void printSamples(const std::vector<std::vector<T>> &samples, const int maxNbSamples, const int maxNbFeatures);
void printLabels(const std::vector<std::size_t> &labels);

char data_filename[128];
char feature_libsvmfilename[128];
int  epoch = 100;

static constexpr const std::size_t Visible = 1323;
static constexpr const std::size_t Hidden = 441;

int main(int argc, char* argv[]) {
	if(argc < 2){
		std::cout << "not enough parameters" << std::endl;
		return 1;
	}

	// feature file path without extension (e.g., .dat)
	// example: parzival_test_21_2000
	sprintf(data_filename, "%s.dat", argv[1]);
	cout << "data file path: " << data_filename << endl;

	sprintf(feature_libsvmfilename, "%s_libsvm", argv[1]);
	cout << "libsvm feature file path: " << feature_libsvmfilename << endl;
	//sprintf(feature_libsvmfilename, "%s_%d_%d_libsvm", argv[1], PATCH_SIZE, NB_PATCHES);

	if(argc > 2) {
		epoch = atoi(argv[2]);
		cout << "#epoch: " << epoch << endl;

		// TODO: get #visible units and #hidden units from your input
	}

	//Call the function you are interested in and complete it

	//read samples with labels
	// std::vector<std::vector<T>> samples;
	// std::vector<std::size_t> labels;
	// read_data(data_filenam, samples, labels);

	//print data
	//printSamples(samples, 10, 30);
	//printLabels(labels);

	// train a rbm and visualize the learned features
	//rbm_features();

	// 1. train rbm with svm
	train_dbn_svm();

	// 2. use the trained svm for classification
	//svm_predict();

	//svm_classify();

    return 0;
}

int doesFileExist(const char *filename) {
    struct stat st;
    int result = stat(filename, &st);
    return result == 0;
}

void printSamples(const std::vector<std::vector<T>> &samples, const std::size_t maxNbSamples, const std::size_t maxNbFeatures) {
	cout << "#samples " << samples.size() << endl;

	auto sampleSize = samples.size();
	if(maxNbSamples > 0) {
		sampleSize = (samples.size() > maxNbSamples) ? maxNbSamples : samples.size();
	}

	for (std::size_t j=0; j<sampleSize; j++) {
		const auto& vec = samples[j];

		auto featureSize = vec.size();
		if(maxNbFeatures > 0) {
			featureSize = (vec.size() > maxNbFeatures) ? maxNbFeatures : vec.size();
		}

		for (std::size_t i=0; i<featureSize; i++) {
			cout << vec[i] << ' ';
		}
		cout << endl;
	}
}

void printLabels(const std::vector<std::size_t> &labels) {
	//std::vector<std::size_t>::const_iterator p1;
	//for (p1 = labels.begin(); p1 != labels.end(); p1++) {
	//		cout << *p1 << ' ';
	//}
	//cout << endl;

	cout << "#labels " << labels.size() << endl;
	for (unsigned int i=0; i<labels.size(); i++) {
		cout << labels[i] << endl;
	}
}

void rbm_features() {
    //1. Configure and create the RBM

	// get #features
	//const int nbFeatures = get_nbcolumns()-1;
    using rbm_t = dll::rbm_desc<
       363,                                           //Number of visible units
       121                                                  //Number of hidden units
       , dll::momentum                                      //Activate momentum ?
       , dll::batch_size<10>                                //Minibatch
       , dll::weight_decay<>                                //Activate weight decay ?
       //, dll::sparsity<>                                  //Activate weight decay ?
       , dll::visible<dll::unit_type::GAUSSIAN>           //Gaussian visible units ?
       , dll::watcher<dll::opencv_rbm_visualizer>         //OpenCV Visualizer ?
    >::rbm_t;

    auto rbm = std::make_unique<rbm_t>();

    //rbm->learning_rate = 0.0001;

    //rbm->load("file.dat"); //Load from file
    //rbm->load(RBM_FILE);

    //2. Read dataset

    std::vector<std::vector<T>> samples;     //All the samples
    std::vector<std::size_t> labels;         //All the labels

    read_data(data_filename, samples, labels);

    //3. Train the RBM for x epochs

    rbm->train(samples, epoch);

    //4. Get the activation probabilities for a sample

    for(auto& sample : samples){
        auto probs = rbm->activation_probabilities(sample);

        //Do something with the extracted features
    }

	//5. Store to file
    //rbm->store("file.dat");
    rbm->store(RBM_FILE);
}

// TODO: get #input units and #hidden units from variables
void train_dbn_svm() {
	//1. Configure and create the RBM

	// get #features. #features is considered as #input units
	// const int nbFeatures = get_nbcolumns(data_filename)-1;

    using dbn_t = dll::dbn_desc<
        dll::dbn_label_layers<
            dll::rbm_desc<Visible, Hidden, dll::batch_size<25>, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
            //dll::rbm_desc<400, 100, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
            //dll::rbm_desc<100, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
        >
        , dll::watcher<dll::opencv_dbn_visualizer> //For visualization
        >::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

	//parameter tunning
    //dbn->layer<0>().learning_rate = 0.01;


    //2. Read dataset

    std::vector<std::vector<T>> samples;     //All the samples
    std::vector<std::size_t> labels;         //All the labels

    read_data(data_filename, samples, labels);


    //3. Train the DBN layers for x epochs

    std::cout << "DBN pretraining ..."  << std::endl;

    dbn->pretrain(samples, epoch);

    //3.1. Get the activation probabilities for a sample
    // create a libsvm format file which contains the features (activation probabilities) of the samples
    std::cout << "Output features to libsvm file ..."  << std::endl;
    remove(feature_libsvmfilename);
    std::ofstream out(feature_libsvmfilename);
    std::cout << "libsvm file: " << feature_libsvmfilename  << std::endl;

	for(std::size_t i = 0; i < samples.size(); ++i) {
		auto& sample = samples[i];
		auto  label = labels[i];

		auto probs = dbn->activation_probabilities(sample);

		// classid 1:value 2:value
		out << label;
		for (std::size_t i=0; i < probs.size(); i++) {
			out << ' ' << (i+1) << ':' << probs[i];
		}
		out << std::endl;
	}
	out.close();

    //for(auto& sample : samples){
    //    auto probs = dbn->activation_probabilities(sample);

		 // do something with the extracted features
        // TODO: save the extracted features to libsvm file format
    //}

    //4. Train the SVM

	std::cout << "SVM pretraining ..."  << std::endl;

	svm_parameter parameters;

	parameters.svm_type = C_SVC;
    parameters.kernel_type = RBF;
    parameters.C = 2.8;
    parameters.gamma = 0.0073;

    parameters.probability = 1;
    parameters.degree = 3;
    parameters.coef0 = 0;
    parameters.nu = 0.5;
    parameters.cache_size = 100;
    parameters.eps = 1e-3;
    parameters.p = 0.1;
    parameters.shrinking = 1;
    parameters.nr_weight = 0;
    parameters.weight_label = nullptr;
    parameters.weight = nullptr;

    dbn->svm_train(samples, labels, parameters);

    //5. Store the DBM and SVM file

    //dbn->store("file.dat"); //Store to file
    std::cout << "save DBN and SVM model to: "  << DBN_SVM_FILE << std::endl;

    dbn->store(DBN_SVM_FILE);

    std::cout << "Done."  << std::endl;
}

void svm_predict() {
	//1. Configure and create the RBM

	// get #features
	//const int nbFeatures = get_nbcolumns()-1;
    using dbn_t = dll::dbn_desc<
        dll::dbn_label_layers<
            dll::rbm_desc<Visible, Hidden, dll::batch_size<25>, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
            //dll::rbm_desc<400, 100, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
            //dll::rbm_desc<100, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
        >
        //, dll::watcher<dll::opencv_dbn_visualizer> //For visualization
        >::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

	// 2. load SVM and DBM model

	dbn->load(DBN_SVM_FILE);


	//3. Read dataset

    std::vector<std::vector<T>> samples;     	   //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(data_filename, samples, labels);

	 //3.1. Get the activation probabilities for a sample

    for(auto& sample : samples){
        auto probs = dbn->activation_probabilities(sample);

        for (std::size_t i=0; i < probs.size(); i++) {
			float feature = probs[i];
		}

        //Do something with the extracted features
    }

	//4. Compute accuracy on the training set

    auto training_error = dll::test_set(dbn, samples, labels, dll::svm_predictor());

    std::cout << "Training error: "  << training_error << std::endl;

	int nbMisClassified = 0;
	for(std::size_t i = 0; i < samples.size(); ++i){
		auto& sample = samples[i];
		auto  label = labels[i];

		auto predicted = dbn->svm_predict(sample);

		if(predicted != label) {
			 std::cout << "mis calssified: "  << i << " predicted: " << predicted << " label: " << label << std::endl;
			 nbMisClassified++;
		}
		//TODO: save prediction results
	}

	std::cout << "#misclassified samples:" << nbMisClassified << endl;
}

void svm_classify() {
    //1. Configure and create the RBM

	// get #features
	//const int nbFeatures = get_nbcolumns()-1;
    using dbn_t = dll::dbn_desc<
        dll::dbn_label_layers<
            dll::rbm_desc<1200, 400, dll::batch_size<10>, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
            //dll::rbm_desc<400, 100, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
            //dll::rbm_desc<100, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
        >
        //, dll::watcher<dll::opencv_dbn_visualizer> //For visualization
        >::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    //dbn->layer<0>().learning_rate = 0.01;

    //dbn->load("file.dat"); //Load from file
    dbn->load(RBM_FILE);

    //2. Read dataset

    std::vector<std::vector<T>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(data_filename, samples, labels);

    //3. Train the DBN layers for x epochs


    //dbn->pretrain(samples, epoch);

    //4. Train the SVM

    //dbn->svm_train(samples, labels);

    //5. Compute accuracy on the training set

    auto training_error = dll::test_set(dbn, samples, labels, dll::svm_predictor());

    std::cout << "Training error: "  << training_error << std::endl;

	for(std::size_t i = 0; i < samples.size(); ++i){
		auto& sample = samples[i];
		auto  label = labels[i];

		auto predicted = dbn->svm_predict(sample);

		//TODO
	}

    //6. Store the file if you want to save it for later

    //dbn->store("file.dat"); //Store to file
    dbn->store(RBM_FILE);
}


/*
 * Read samples and labels
 *
 * file format:
 * class_id, feature value_1, feature value_2, feature value_3, ..., feature value_n
 */
void read_data(char* data_filename, std::vector<std::vector<T>>& samples, std::vector<std::size_t>& labels) {

	//if(access(DATA_FILE, F_OK) != 1){
	if (doesFileExist(data_filename)){

		std::cout << "reading " << data_filename  << " ..." << std::endl;

		//Open file
		FILE* asc = fopen(data_filename, "r");

		//get number of lines and columns
		int nb_lines, nb_columns = 0;
		nb_lines = get_nblines(data_filename);
		nb_columns = get_nbcolumns(data_filename);
		std::cerr << "Nb lines:" << nb_lines << std::endl;
		std::cerr << "Nb columns:" << nb_columns << std::endl;
		//printf("number of lines in %s = %d \n", DATA_FILE, nb_lines-1);
		//printf("number of columns in %s = %d \n", DATA_FILE, nb_columns);

		//read samples
		for (int l=0; l < nb_lines; l++) {

			// read class id
			std::size_t classNum = 0;
			fscanf(asc, "%lu", &classNum);
			labels.push_back(classNum);

			// read features
			std::vector<T> col;
			for (int col_num=0; col_num < nb_columns-1; col_num++) {
				T lf = -1;
				fscanf(asc, READ_FEATURE_FORMAT, &lf);
				col.push_back(lf);
			}

			samples.push_back(col);
		}

		fclose(asc);

		std::cout << "Done. " << std::endl;

		std::cout << "Normalizing. " << std::endl;
		//normalize features to have 0 mean and 1 variance
		cpp::normalize_each(samples); //For gaussian visible units

		std::cout << "Done. " << std::endl;
	} else {
		std::cout << DATA_FILE  << " does not exist." << std::endl;
	}
}

int get_nblines(char* filename) {
	FILE* myfile = fopen(filename, "r");
	int ch, number_of_lines = 0;

	do {
		ch = fgetc(myfile);
		if(ch == '\n')
			number_of_lines++;
	} while (ch != EOF);

	// last line doesn't end with a new line!
	// but there has to be a line at least before the last line
	if(ch != '\n' && number_of_lines != 0)
		number_of_lines++;

	fclose(myfile);

	//printf("number of lines in test.txt = %d", number_of_lines);
	return number_of_lines-1;
}

int get_nbcolumns(char* filename) {
	FILE* myfile = fopen(filename, "r");
	int ch, number_of_columns = 0;

	do {
		ch = fgetc(myfile);
		if(ch == ',') {
			//std::cerr << ", -> " << number_of_columns << std::endl;
			number_of_columns++;
		} else {
			//std::cerr << (char)ch;
		}
	} while (ch != EOF &&  ch != '\n');

	fclose(myfile);

	//printf("number of lines in test.txt = %d", number_of_lines);
	return number_of_columns+1;
}
