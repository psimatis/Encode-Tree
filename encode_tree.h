#include <torch/torch.h>
#include <torch/csrc/api/include/torch/data/dataloader.h>
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

typedef tlx::btree_multimap<key_type, value_type, less<key_type>, btree_map_traits<key_type,value_type>, allocator<pair<key_type, value_type> > > btree_mm;

class encode_tree{
    public:
        btree_mm btree;
        vector<element> datam;
        int dimensions;
        float tree_build_time, index_build_time;
        AEImpl* model;


    encode_tree(int dimensions, int hSize, int codeSize, int depth){
        model = new AEImpl(dimensions, hSize, codeSize, depth);
    }

    void train(int epochs, float learning_rate, int batchSize, string path){
        auto dataset = CustomDataset(path).map(torch::data::transforms::Stack<>());
        auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset), batchSize);
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
        for (size_t epoch = 0; epoch != epochs; ++epoch) {
            double epochLoss = 0;
            torch::Tensor loss;
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
    }

    void build_index(string path){
        auto dataset = CustomDataset(path).map(torch::data::transforms::Stack<>());
        auto encodeLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(move(dataset), 1);
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
    }

    vector<element> range_query(torch::Tensor low_t, torch::Tensor high_t){
        float low = model->encode(low_t).item<double>();
        float high = model->encode(high_t).item<double>();
        leaf_count = 1;
        vector<element> res;
        vector<element> candidate_set;
        btree_mm::iterator bi;

        
        bi = btree.lower_bound(low);

        while(bi != btree.end() && bi.key() <= high){
            candidate_set.push_back((*bi).second);
            bi++;
        }

        for (auto c: candidate_set){
            if (overlaps(c.point, low_t, high_t))
                res.push_back(c);
        }
        cout << leaf_count << " " <<  res.size() << endl;
        return res;
    }

    bool overlaps(torch::Tensor p, torch::Tensor lowerCorner, torch::Tensor upperCorner){
        for(long dim = 0; dim < p.sizes()[1]; dim++){
            if (p.index({0,dim}).item<float>() < lowerCorner.index({0,dim}).item<float>() || p.index({0,dim}).item<float>() > upperCorner.index({0,dim}).item<float>())
                return false;
        }
        return true;
    }

};
