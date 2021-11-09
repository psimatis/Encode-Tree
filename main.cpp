#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "CustomLoaders.h"
#include "autoencoder.h"
#include "tlx/container/btree_multimap.hpp"
#include "encode_tree.h"

using namespace std;
using namespace torch::indexing;

int main() {
    // Hyper parameters
    int seed = 42;
    torch::manual_seed(seed);
    const int64_t hSize = 2;
    const int64_t codeSize = 1;
    const int64_t dimensions = 4;
    const int64_t batchSize = 1024;
    const size_t epochs = 3;
    const int depth = 2;
    const double learning_rate = 0.001;
    string path = "../data/4D-1e6_norm.csv";

	// Data
 //    auto dataset = CustomDataset("../data/4D-1e6_norm.csv").map(torch::data::transforms::Stack<>());
 //    auto dataset1 = CustomDataset("../data/4D-1e6_norm.csv").map(torch::data::transforms::Stack<>());
	// auto num_samples = dataset.size().value();
	// cout << "Dataset size:" << dataset.size().value() << endl;
	// auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset), batchSize);
	// auto encodeLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset1), 1);

	//Queries
    auto qSet = CustomQueryset("../data/4D-qI0_norm_1000").map(torch::data::transforms::Stack<>());
	auto numQueries = qSet.size().value();
	cout << "Queryset size:" << qSet.size().value() << endl;
	auto qLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(qSet), 1);


    // Model
	encode_tree tree(dimensions, hSize, codeSize, depth);
	cout << "Con done" << endl;
    // Train the model
    tree.train(epochs, learning_rate, batchSize, path);
    cout << "Train done" << endl;
    // Encode data
    tree.build_index(path);
    cout << "Build done" << endl;
	// Queries
	for (auto& batch: *qLoader){
		tree.range_query(batch.data, batch.target);
	}
}
