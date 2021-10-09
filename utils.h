#pragma once

#include <vector>
#include <torch/torch.h>

using namespace std;

vector<vector<float>> readFile(string fileName){
    vector<vector<float>> data;
    string line;
    float x,y;
    ifstream f(fileName);
    if (f.is_open()){
        while(getline(f,line)){
            istringstream buf(line);
            buf >> x >> y;
            vector<float> point = {x,y};
            data.emplace_back(point);
        }
        f.close();
    } else cout << "cant read file" << fileName << endl;
    return data;
}

float getReward(vector<float> q, vector<float> l){
    float x = q[0] - l[0];
    float y = q[1] - l[1];
    return sqrt(x*x + y*y);
}

vector<float> normalize(vector<float> q){
    vector<float> nq = {0,0};
    nq[0] = (q[0]-25)/50;
    nq[1] = (q[1]+35)/80;
    return nq;
}

bool isAccurate(vector<vector<float>> landmarks, int index, vector<float> q){
    int lgt;
    float minDist = 999999;
    for (int i = 0; i < landmarks.size(); i++){ 
        float dist = getReward(q, landmarks[i]); 
        if (dist < minDist){
            minDist = dist;
            lgt = i;
        }
    }
    if (index == lgt) return true;
    return false;
}

tuple<vector<int>, vector<int>, int> getLandmarks(vector<vector<float>> landmarks, vector<vector<float>> queries, NeuralNet model){
    vector<int> p_indexes, a_indexes;
    int correct = 0;
    tuple<vector<int>, vector<int>, int> T;
    for (vector<float> q: queries){
        auto nq = normalize(q);
        auto data = torch::from_blob(nq.data(), {1,2});
        auto output = model->forward(data);
        auto p_index = get<1>(output.min(1)).item<int>();
        vector<float> l = landmarks[p_index];
        auto min_pred = get<0>(output.min(1));
        int a_index;
        float minDist = 999999;
        for (int i = 0; i < landmarks.size(); i++){
            float dist = getReward(q, landmarks[i]);
            if (dist < minDist) {
                minDist = dist;
                a_index = i;
            }
        }
        if (p_index == a_index) correct++;
        p_indexes.push_back(p_index);
        a_indexes.push_back(a_index);
    }
    T = make_tuple(p_indexes, a_indexes, correct);
    return T;
}

int countCorrect(vector<vector<float>> landmarks, vector<vector<float>> queries, NeuralNet model){
    tuple<vector<int>, vector<int>, int> T = getLandmarks(landmarks, queries, model);
    auto correct = get<2>(T);
   return correct;
}

void printOutcome(vector<vector<float>> landmarks, vector<vector<float>> queries, NeuralNet model){
    tuple<vector<int>, vector<int>, int> T = getLandmarks(landmarks, queries, model);
    auto p_indexes = get<0>(T);
    auto a_indexes = get<1>(T);
    for (int i = 0; i < queries.size(); i++){
        cout << queries[i][0] << ", " << queries[i][1] << ", " << p_indexes[i] << ", " << a_indexes[i] << endl;
    }
}

void printGrads(NeuralNet model){
    cout <<"in weight:" << model->in->weight.grad() << endl;
    cout <<"in bias:" << model->in->bias.grad() << endl;
    //cout <<"h weight:" << model->fcs[0]->weight.grad() << endl;
    //cout <<"h bias:" << model->fcs[0]->bias.grad() << endl;
    cout << "out weight:" << model->out->weight.grad() << endl;
    cout << "out bias:" << model->out->bias.grad() << endl;        
}

void printParameters(NeuralNet model){
    for (const auto& p : model->parameters()) cout << p << endl;
}
