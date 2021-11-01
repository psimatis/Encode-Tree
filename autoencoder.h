#pragma once

#include <torch/torch.h>
#include <utility>

using namespace std;

class VAEImpl : public torch::nn::Module {
 public:
    VAEImpl(int64_t image_size, int64_t h_dim, int64_t z_dim);
    torch::Tensor decode(torch::Tensor z);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor encode(torch::Tensor x);

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
    torch::nn::Linear fc5;
    torch::nn::Linear fc6;
    torch::nn::Linear fc8;
    torch::nn::Linear fc7;
};

TORCH_MODULE(VAE);
