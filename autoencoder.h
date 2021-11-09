#pragma once

#include <torch/torch.h>

using namespace std;

class AEImpl : public torch::nn::Module {
   public:
      AEImpl(int64_t image_size, int64_t h_dim, int64_t z_dim, int depth1);
      torch::Tensor decode(torch::Tensor z);
      torch::Tensor forward(torch::Tensor x);
      torch::Tensor encode(torch::Tensor x);

      int depth;
      vector<torch::nn::Linear> fcs;
};

TORCH_MODULE(AE);
