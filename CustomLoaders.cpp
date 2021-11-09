#include "CustomLoaders.h"

using namespace std;

// Data
torch::Tensor read_data(const string& path) {
   	ifstream data;
   	data.open(path, ios_base::in);
   	if (!data.is_open()) cout << path << " not found!" << endl;

   	int dim = -1;
   	vector<float> inputs = process_data(data, dim);
   	auto input_tensors = torch::from_blob(inputs.data(), {int(inputs.size()/dim), dim}).clone();

   	return input_tensors;
}

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

// Queries
pair<torch::Tensor, torch::Tensor> read_queries(const string& path) {
   	ifstream data;
   	data.open(path, ios_base::in);
   	if (!data.is_open()) cout << path << " not found!" << endl;

   	int dim = -1;
   	vector<float> q = process_queries(data, dim);
    vector<float> low, high;
    for (size_t i = 0; i < q.size(); i += 2*dim){
        low.insert(low.end(), q.begin() + i, q.begin() + i + dim);
        high.insert(high.end(), q.begin() + i + dim, q.begin() + i + 2*dim);
    }
   	auto lowT = torch::from_blob(low.data(), {int(low.size()/dim), dim}).clone();
   	auto highT = torch::from_blob(high.data(), {int(high.size()/dim), dim}).clone();

   	return {lowT, highT};
}

CustomQueryset::CustomQueryset(const string& path) {
    auto q = move(read_queries(path));
    lowT_ = move(q.first);
    highT_ = move(q.second);
}

torch::data::Example<> CustomQueryset::get(size_t index) {
    return {lowT_[index], highT_[index]};
}

torch::optional<size_t> CustomQueryset::size() const {
    return lowT_.size(0);
}
