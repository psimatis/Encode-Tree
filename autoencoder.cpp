#include "autoencoder.h"
#include <utility>

using namespace std;

VAEImpl::VAEImpl(int64_t inputSize, int64_t hSize, int64_t codeSize)
    : fc1(inputSize, hSize), fc2(hSize, hSize), fc3(hSize, hSize), fc4(hSize, codeSize), 
    fc5(codeSize, hSize), fc6(hSize, hSize), fc7(hSize, hSize), fc8(hSize, inputSize) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    register_module("fc5", fc5);
    register_module("fc6", fc6);
    register_module("fc7", fc7);
    register_module("fc8", fc8);
}

torch::Tensor VAEImpl::encode(torch::Tensor x) {
    auto h = torch::nn::functional::relu(fc1->forward(x));
    auto h2 = torch::nn::functional::relu(fc2->forward(h));
    auto h3 = torch::nn::functional::relu(fc3->forward(h2));
    return fc4->forward(h3);
}

torch::Tensor VAEImpl::decode(torch::Tensor z) {
    auto h = torch::nn::functional::relu(fc5->forward(z));
    auto h2 = torch::nn::functional::relu(fc6->forward(h));
    auto h3 = torch::nn::functional::relu(fc7->forward(h2));
    return torch::relu(fc8->forward(h3));
}

torch::Tensor VAEImpl::forward(torch::Tensor x) {
    auto codes = encode(x);
    return decode(codes);
}
