#include "CustomDataset.h"

using namespace std;

namespace {

	torch::Tensor read_data(const string& path) {
    	ifstream data;
    	data.open(path, ios_base::in);
    	if (!data.is_open()) cout << path << " not found!" << endl;

    	int dim = -1;
    	vector<float> inputs = process_data(data, dim);
    	auto input_tensors = torch::from_blob(inputs.data(), {int(inputs.size()/dim), dim}).clone();

    	return input_tensors;
	}
}  // namespace

CustomDataset::CustomDataset(const string& path) {
    input_tensors_ = move(read_data(path));
}

torch::data::Example<> CustomDataset::get(size_t index) {
    return {input_tensors_[index], input_tensors_[index]};
}

torch::optional<size_t> CustomDataset::size() const {
    return input_tensors_.size(0);
}

const torch::Tensor& CustomDataset::features() const {
    return input_tensors_;
}
