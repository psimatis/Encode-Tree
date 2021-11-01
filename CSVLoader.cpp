#include "CSVLoader.h"

using namespace std;

float CSVRow::operator[](size_t index){
    string& eg = m_data[index];
    return atof(eg.c_str());
}

size_t CSVRow::size() const{
    return m_data.size();
}

void CSVRow::readNextRow(istream& str) {
    string line, cell;
    
    getline(str, line);
    stringstream lineStream(line);
    
    m_data.clear();
    while (getline(lineStream, cell, ','))
        m_data.push_back(cell);
        
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty()){
        // If there was a trailing comma then add an empty element.
        m_data.push_back("");
    }
}

istream& operator>>(istream& str, CSVRow& data){
	data.readNextRow(str);
	return str;
}

vector<float> process_data(ifstream& file) {
	vector<vector<float>> features;

	CSVRow  row;
	while (file >> row) {
		features.emplace_back();
        // skips first column (id)
		for (size_t loop = 1;loop < row.size(); ++loop)
			features.back().emplace_back(row[loop]);
	}
	// Flatten features vectors to 1D
	vector<float> inputs = features[0];
	int64_t total = accumulate(begin(features) + 1, end(features), 0UL, [](size_t s, vector<float> const& v){return s + v.size();});

	inputs.reserve(total);
	for (size_t i = 1; i < features.size(); i++)
		inputs.insert(inputs.end(), features[i].begin(), features[i].end());
	
	return inputs;
}
