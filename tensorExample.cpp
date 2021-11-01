// Tutorial from: https://radicalrafi.github.io/posts/pytorch-cpp-intro/
#include<bits/stdc++.h>
#include<torch/torch.h>

using namespace std;

int main(){
    vector<float> vec = {0.2, 0.9, 0.1, 0.25, 0.006, 0.01};
	// auto a = torch::tensor(data).reshape({1,2});
	auto a = torch::from_blob(vec.data(), {int(vec.size()/2),2});
	// auto a = torch::tensor(data).view({1,2});
	cout << a << endl;
	cout << a.options() << endl;
}
