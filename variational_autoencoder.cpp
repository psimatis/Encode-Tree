#include "variational_autoencoder.h"
#include <utility>

using namespace std;

VAEImpl::VAEImpl(int64_t inputSize, int64_t hSize, int64_t codeSize)
    : fc1(inputSize, hSize), fc2(hSize, codeSize), fc3(codeSize, hSize), fc4(hSize, inputSize) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
}

torch::Tensor VAEImpl::encode(torch::Tensor x) {
    auto h = torch::nn::functional::relu(fc1->forward(x));
    return fc2->forward(h);
}

torch::Tensor VAEImpl::decode(torch::Tensor z) {
    auto h = torch::nn::functional::relu(fc3->forward(z));
    return torch::sigmoid(fc4->forward(h));
}

VAEOutput VAEImpl::forward(torch::Tensor x) {
    auto codes = encode(x);
    auto decoded = decode(codes);
    return {decoded};
}
