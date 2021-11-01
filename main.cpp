#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "CustomDataset.h"
#include "autoencoder.h"

using namespace std;

int main() {
    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    int seed = 2;
    torch::manual_seed(seed);
    const int64_t hSize = 2;
    const int64_t codeSize = 1;
    const int64_t inputSize = 4;
    const int64_t batchSize = 1024;
    const size_t epochs = 3;
    const double learning_rate = 1e-3;

	// Data
    auto dataset = CustomDataset("../data/4D-1e6_norm.csv").map(torch::data::transforms::Stack<>());
	auto num_samples = dataset.size().value();
	cout << "Dataset size:" << dataset.size().value() << endl;
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset), batchSize);

    // Model
    VAE model(inputSize, hSize, codeSize);
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Train the model
    for (size_t epoch = 0; epoch != epochs; ++epoch) {
        torch::Tensor input, output;
        double epochLoss = 0;;
        size_t batch_index = 0;
        model->train();

        for (auto& batch : *dataloader) {
            input = batch.data.to(device);
           	output = model->forward(input);
			auto loss = torch::nn::functional::mse_loss(output, input);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
  
            if ((batch_index + 1) % 100 == 0) {
                cout << "Epoch [" << epoch + 1 << "/" << epochs
                	<< "], Step [" << batch_index + 1 << "/" << num_samples / batchSize
                	<< "], Loss: " << loss.item<double>() / batch.data.size(0) << "\n";
            }
            batch_index++;
            epochLoss += loss.item<double>();
        }
        cout << "Epoch: " << epoch << ", Loss: " << epochLoss << endl;  

        model->eval();
        torch::NoGradGuard no_grad;
    }
	cout << "Training finished!\n";
}
