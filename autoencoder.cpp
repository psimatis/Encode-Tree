#include "autoencoder.h"

using namespace std;

AEImpl::AEImpl(int64_t inputSize, int64_t hSize, int64_t codeSize, int depth1) { 
	depth = depth1;
	int layer = 0; 
	int i = 0;
	string l;
	
	fcs.push_back(register_module("fc0", torch::nn::Linear(inputSize, hSize)));
	layer++;
	for (i = layer; i < layer + depth; i++){
		l = "fc" + to_string(i);
		fcs.push_back(register_module(l, torch::nn::Linear(hSize, hSize)));
	}
	layer += depth;

	l = "fc" + to_string(layer);
	fcs.push_back(register_module(l, torch::nn::Linear(hSize, codeSize)));
	layer++;
	
	l = "fc" + to_string(layer);
	fcs.push_back(register_module(l, torch::nn::Linear(codeSize, hSize)));

	for (i = layer + 1; i < layer + depth; i++){
		l = "fc" + to_string(i);
		fcs.push_back(register_module(l, torch::nn::Linear(hSize, hSize)));
	}
	layer += depth;
	
	l = "fc" + to_string(layer);
    fcs.push_back(register_module(l, torch::nn::Linear(hSize, inputSize)));   
}

torch::Tensor AEImpl::encode(torch::Tensor x) {
	auto h = x;
	for (int i = 0; i < depth + 1; i++)
    	h = torch::nn::functional::relu(fcs[i]->forward(h));
    return fcs[depth + 1]->forward(h);
}

torch::Tensor AEImpl::decode(torch::Tensor z) {
    auto h = z;
    for (int i = depth + 2; i < fcs.size() - 1; i++)
    	h = torch::nn::functional::relu(fcs[i]->forward(h));
    return fcs[fcs.size() - 1]->forward(h);
}

torch::Tensor AEImpl::forward(torch::Tensor x) {
    auto codes = encode(x);
    return decode(codes);
}
