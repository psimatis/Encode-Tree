#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "CustomDataset.h"
#include "variational_autoencoder.h"

using namespace std;

int main() {
    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t h_dim = 400;
    const int64_t z_dim = 20;
    const int64_t input_size = 4;
    const int64_t batch_size = 100;
    const size_t num_epochs = 3;
    const double learning_rate = 1e-3;

	// Data
    // auto dataset = torch::data::datasets::MNIST(MNIST_data_path).map(torch::data::transforms::Stack<>());
    auto dataset = CustomDataset("../data/4D-1e6_norm.csv").map(torch::data::transforms::Stack<>());
	auto num_samples = dataset.size().value();
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset), batch_size);

    // Model
    VAE model(input_size, h_dim, z_dim);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    cout << fixed << setprecision(4);

    cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        torch::Tensor images;
        size_t batch_index = 0;

        model->train();

        for (auto& batch : *dataloader) {
            // Transfer images to device
            images = batch.data.reshape({-1, input_size}).to(device);

            // Forward pass
            auto output = model->forward(images);
	        auto reconstruction_loss = torch::nn::functional::binary_cross_entropy(output.reconstruction, images,
                torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(torch::kSum));

            // Backward pass and optimize
            auto loss = reconstruction_loss;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            if ((batch_index + 1) % 100 == 0) {
                cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
                    << num_samples / batch_size << "], Reconstruction loss: "
                    << reconstruction_loss.item<double>() / batch.data.size(0) << "\n";
            }
            ++batch_index;
        }

        model->eval();
        torch::NoGradGuard no_grad;\
    }
	cout << "Training finished!\n";
}
