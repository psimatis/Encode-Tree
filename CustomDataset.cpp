#include "CustomDataset.h"

using namespace std;

namespace {
	constexpr uint32_t kTrainSize = 800;
	constexpr uint32_t kTestSize = 200;
	constexpr uint32_t kSizePerBatch = 100;
	constexpr uint32_t kFeature = 2;
	constexpr uint32_t kBytesPerBatchFile = (kFeature+1) * kSizePerBatch;

	pair<torch::Tensor, torch::Tensor> read_data(const string& path, bool train) {
    	ifstream data;
    	data.open(path, ios_base::in);
    	if (!data.is_open()) cout << path << " not found!" << endl;
    
    	pair<vector<float>, vector<float>> out = process_data(data);
    	vector<float> inputs = out.first;
    	vector<float> outputs = out.second;
    	auto output_tensors = torch::from_blob(outputs.data(), {int(outputs.size()), 1}).clone();
    	auto input_tensors = torch::from_blob(inputs.data(), {int(outputs.size()), int(inputs.size()/outputs.size())}).clone();

    	// cout << input_tensors.to(torch::kFloat32) << endl << "output: " << output_tensors.to(torch::kInt64) << endl;
    	return {input_tensors.to(torch::kFloat32), output_tensors.to(torch::kInt64)};
	}
}  // namespace

CustomDataset::CustomDataset(const string& path, Mode mode) : mode_(mode) {
    auto data = read_data(path, mode == Mode::kTrain);
    input_tensors_ = move(data.first);
    output_tensors_ = move(data.second);
}

torch::data::Example<> CustomDataset::get(size_t index) {
    return {input_tensors_[index], output_tensors_[index]};
}

torch::optional<size_t> CustomDataset::size() const {
    return output_tensors_.size(0);
}

bool CustomDataset::is_train() const noexcept {
    return mode_ == Mode::kTrain;
}

const torch::Tensor& CustomDataset::features() const {
    return input_tensors_;
}

const torch::Tensor& CustomDataset::labels() const {
    return output_tensors_;
}
