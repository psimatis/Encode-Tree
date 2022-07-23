#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "CustomLoaders.h"
#include "autoencoder.h"
#include "tlx/container/btree_multimap.hpp"
#include "encode_tree.h"
#include <bits/stdc++.h>

using namespace std;
using namespace torch::indexing;
using namespace std::chrono;

int main() {
    // Hyper parameters
    int seed = 42;
    torch::manual_seed(seed);
    const int64_t hSize = 2;
    const int64_t codeSize = 1;
    const int64_t dimensions = 4;
    const int64_t batchSize = 1024;
    const size_t epochs = 3;
    const double learning_rate = 0.001;
    string path = "...";

	//Queries
    auto qSet = CustomQueryset("...").map(torch::data::transforms::Stack<>());
	auto numQueries = qSet.size().value();
	cout << "Queryset size:" << qSet.size().value() << endl;
	auto qLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(move(qSet), 1);

    // Model
	encode_tree tree(dimensions, hSize, codeSize);
    // Train the model
    tree.train(epochs, learning_rate, batchSize, path);
    cout << "Train done" << endl;
    // Encode data
    tree.build_index(path);
    cout << "Build done" << endl;
	// Queries
	for (auto& batch: *qLoader){
		tree.range_query(batch.data, batch.target);
        tree.seq_range_query(batch.data, batch.target);
	}
}
