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
    const int64_t hSize = 2;
    const int64_t codeSize = 1;
    const int64_t inputSize = 4;
    const int64_t batchSize = 1024;
    const size_t epochs = 1;
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

    // Set floating point output precision
    cout << fixed << setprecision(4);

    // Train the model
    for (size_t epoch = 0; epoch != epochs; ++epoch) {
        torch::Tensor inputs;
        size_t batch_index = 0;
        model->train();

        for (auto& batch : *dataloader) {
            // Transfer images to device
            inputs = batch.data.reshape({-1, inputSize}).to(device);

            // Forward pass
            auto output = model->forward(inputs);
			auto loss = torch::nn::functional::mse_loss(output, inputs);

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            if ((batch_index + 1) % 100 == 0) {
                cout << "Epoch [" << epoch + 1 << "/" << epochs 
                	<< "], Step [" << batch_index + 1 << "/" << num_samples / batchSize 
                	<< "], Loss: " << loss.item<double>() / batch.data.size(0) << "\n";
            }
            batch_index++;
        }

        model->eval();
        torch::NoGradGuard no_grad;
    }
	cout << "Training finished!\n";
}
