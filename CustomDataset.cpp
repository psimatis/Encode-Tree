#include "CustomDataset.h"

using namespace std;

namespace {

	torch::Tensor read_data(const string& path, bool train) {
    	ifstream data;
    	data.open(path, ios_base::in);
    	if (!data.is_open()) cout << path << " not found!" << endl;
    
    	vector<float> inputs = process_data(data);
    	auto input_tensors = torch::from_blob(inputs.data(), {int(inputs.size()/4), 4}).clone();

    	// cout << input_tensors << endl;
    	//cout << input_tensors.sizes()[0] << "," << input_tensors.sizes()[1] << endl;
    	return input_tensors;
	}
}  // namespace

CustomDataset::CustomDataset(const string& path, Mode mode) : mode_(mode) {
    input_tensors_ = move(read_data(path, mode == Mode::kTrain));
}

torch::data::Example<> CustomDataset::get(size_t index) {
    return {input_tensors_[index], input_tensors_[index]};
}

torch::optional<size_t> CustomDataset::size() const {
    return input_tensors_.size(0);
}

bool CustomDataset::is_train() const noexcept {
    return mode_ == Mode::kTrain;
}

const torch::Tensor& CustomDataset::features() const {
    return input_tensors_;
}
