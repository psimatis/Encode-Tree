#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "CustomDataset.h"
#include "autoencoder.h"

using namespace std;

int main() {
    // Hyper parameters
    int seed = 42;
    torch::manual_seed(seed);
    const int64_t hSize = 4;
    const int64_t codeSize = 1;
    const int64_t inputSize = 4;
    const int64_t batchSize = 1024;
    const size_t epochs = 3;
    const double learning_rate = 0.001;

	// Data
    auto dataset = CustomDataset("../data/4D-1e6_norm.csv").map(torch::data::transforms::Stack<>());
	auto num_samples = dataset.size().value();
	cout << "Dataset size:" << dataset.size().value() << endl;
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset), batchSize);

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
  
            // if ((batch_index + 1) % 100 == 0) { // prints every few batches
                // cout << "Epoch [" << epoch << "/" << epochs
                	// << "], Step [" << batch_index << "/" << num_samples / batchSize
                	// << "], Loss: " << loss.item<double>() / batch.data.size(0) << "\n";
            // }
            // batch_index++;
            epochLoss += loss.item<double>();
        }
        cout << "Epoch: " << epoch << ", Loss: " << epochLoss << endl;  
        cout << "Epoch: " << epoch << ", Loss: " << loss.item<double>() << endl;  

        model->eval();
        torch::NoGradGuard no_grad;
    }
	cout << "Training finished!\n";
}
