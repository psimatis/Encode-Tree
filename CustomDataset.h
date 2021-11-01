#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <string>
#include <CSVLoader.h>

using namespace std;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
 public:
    // The mode in which the dataset is loaded
    enum Mode {kTrain, kTest};

    explicit CustomDataset(const string& root, Mode mode = Mode::kTrain);

    // Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) override;

    // Returns the size of the dataset.
    torch::optional<size_t> size() const override;

    // Returns true if this is the training subset of LMD.
    bool is_train() const noexcept;

    // Returns all images stacked into a single tensor.
    const torch::Tensor& features() const;

    // Returns all targets stacked into a single tensor.
    const torch::Tensor& labels() const;

 private:
    torch::Tensor input_tensors_;
    torch::Tensor output_tensors_;
    Mode mode_;
};
