#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "CustomLoaders.h"
#include "autoencoder.h"
#include "tlx/container/btree_multimap.hpp"

using namespace std;
using namespace torch::indexing;

template <typename _Key, typename _Data>
struct btree_map_traits
{
    static const bool   self_verify = false;
    static const bool   debug = false;
    static const int    leaf_slots = TLX_BTREE_MAX( 8, 4*1024 / (sizeof(_Key) + sizeof(float) * 4));
    static const int    inner_slots = TLX_BTREE_MAX( 8, 4*1024 / (sizeof(_Key) + sizeof(void*)));
    static const int binsearch_threshold = 256;
};

struct element {
	string label;
	torch::Tensor point;
};

typedef float key_type;
typedef element value_type;

typedef tlx::btree_multimap<key_type, value_type, less<key_type>,
        btree_map_traits<key_type,value_type>, allocator<pair<key_type, value_type> > > btree_mm;


vector<element> rangeQuery(btree_mm btree, float min, float max){
	btree_mm::iterator bi;

	vector<element> candidateSet;
	bi = btree.lower_bound(min);

	while(bi != btree.end() && bi.key() <= max){
		candidateSet.push_back((*bi).second);
		bi++;
	}
	return candidateSet;
}

bool overlaps(torch::Tensor p, torch::Tensor lowerCorner, torch::Tensor upperCorner){
        for(long dim = 0; dim < p.sizes()[1]; dim++){
            if (p.index({0,dim}).item<float>() < lowerCorner.index({0,dim}).item<float>() || p.index({0,dim}).item<float>() > upperCorner.index({0,dim}).item<float>())
                return false;
        }
        return true;
}

int main() {

	btree_mm btree;

    // Hyper parameters
    int seed = 42;
    torch::manual_seed(seed);
    const int64_t hSize = 4;
    const int64_t codeSize = 1;
    const int64_t inputSize = 4;
    const int64_t batchSize = 1024;
    const size_t epochs = 3;
    const double learning_rate = 0.001;

	// Data
    auto dataset = CustomDataset("../data/4D-1e6_norm.csv").map(torch::data::transforms::Stack<>());
    auto dataset1 = CustomDataset("../data/4D-1e6_norm.csv").map(torch::data::transforms::Stack<>());
	auto num_samples = dataset.size().value();
	cout << "Dataset size:" << dataset.size().value() << endl;
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset), batchSize);
	auto encodeLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset1), 1);

	//Queries
    auto qSet = CustomQueryset("../data/4D-qI0_norm_1000").map(torch::data::transforms::Stack<>());
	auto numQueries = qSet.size().value();
	cout << "Queryset size:" << qSet.size().value() << endl;
	auto qLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(qSet), 1);

    // Model
    AE model(inputSize, hSize, codeSize);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Train the model
    for (size_t epoch = 0; epoch != epochs; ++epoch) {
        double epochLoss = 0;
        torch::Tensor loss;
        size_t batch_index = 0;
        model->train();

        for (auto& batch : *dataloader) {
            auto input = batch.data;
           	auto output = model->forward(input);
			loss = torch::nn::functional::mse_loss(output, input);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            epochLoss += loss.item<double>();
        }
        cout << "Epoch: " << epoch << ", Loss: " << epochLoss << endl;  

        model->eval();
        torch::NoGradGuard no_grad;
    }

    // Encode data
    vector<pair<float, element>> bulk_data;
    for (auto& batch: *encodeLoader){
    	float code = model->encode(batch.data).item<double>();
	element el;
	el.label = "a";
	el.point = batch.data;
    	bulk_data.push_back(make_pair(code, el));
    }
	sort(bulk_data.begin(), bulk_data.end(), [] (const pair<float, element> &x, const pair<float, element> &y) {return x.first < y.first;});
    btree.bulk_load(bulk_data.begin(), bulk_data.end());
	cout << "Finished bulk loading" << endl;
	// Queries
	for (auto& batch: *qLoader){
		auto q = batch.data;
		float low = model->encode(batch.data).item<double>();
		float high = model->encode(batch.target).item<double>();
		leaf_count = 1;
		auto candidate_set = rangeQuery(btree, low, high);
		vector<element> res;
		for (auto c: candidate_set){
			if (overlaps(c.point, batch.data, batch.target))
				res.push_back(c);
		}
		cout << leaf_count << " " <<  res.size() << endl;
	}
}
