#include "autoencoder.h"

using namespace std;

AEImpl::AEImpl(int64_t inputSize, int64_t hSize, int64_t codeSize)
    : fc1(inputSize, hSize), fc2(hSize, hSize), fc3(hSize, codeSize), 
    fc4(codeSize, hSize), fc5(hSize, hSize), fc6(hSize, inputSize) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    register_module("fc5", fc5);
    register_module("fc6", fc6);
}

torch::Tensor AEImpl::encode(torch::Tensor x) {
    auto h = torch::nn::functional::relu(fc1->forward(x));
    auto h2 = torch::nn::functional::relu(fc2->forward(h));
    return fc3->forward(h2);
}

torch::Tensor AEImpl::decode(torch::Tensor z) {
    auto h = torch::nn::functional::relu(fc4->forward(z));
    auto h2 = torch::nn::functional::relu(fc5->forward(h));
    return fc6->forward(h2);
}

torch::Tensor AEImpl::forward(torch::Tensor x) {
    auto codes = encode(x);
    return decode(codes);
}
