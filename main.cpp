#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "CustomLoaders.h"
#include "autoencoder.h"

using namespace std;
using namespace torch::indexing;

int main() {
    // Hyper parameters
    int seed = 42;
    torch::manual_seed(seed);
    const int64_t hSize = 4;
    const int64_t codeSize = 1;
    const int64_t inputSize = 4;
    const int64_t batchSize = 1024;
    const size_t epochs = 1;
    const double learning_rate = 0.001;

	// Data
    auto dataset = CustomDataset("../data/4D-1e6_norm.csv").map(torch::data::transforms::Stack<>());
	auto num_samples = dataset.size().value();
	cout << "Dataset size:" << dataset.size().value() << endl;
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset), batchSize);

	//Queries
    auto qSet = CustomQueryset("../data/4D-qI0_norm").map(torch::data::transforms::Stack<>());
	auto numQueries = qSet.size().value();
	cout << "Queryset size:" << qSet.size().value() << endl;
	auto qLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(qSet), 1);

    // Model
    AE model(inputSize, hSize, codeSize);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Train the model
    for (size_t epoch = 0; epoch != epochs; ++epoch) {
        double epochLoss = 0;
        torch::Tensor loss;
        size_t batch_index = 0;
        model->train();

        for (auto& batch : *dataloader) {
            auto input = batch.data;
           	auto output = model->forward(input);
			loss = torch::nn::functional::mse_loss(output, input);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            epochLoss += loss.item<double>();
        }
        cout << "Epoch: " << epoch << ", Loss: " << epochLoss << endl;  

        model->eval();
        torch::NoGradGuard no_grad;
    }

	// Queries
	for (auto& batch: *qLoader){
		auto q = batch.data;
		auto low = model->encode(batch.data);
		auto high = model->encode(batch.target);
		cout << low << high << endl;

		break;
	}
}
