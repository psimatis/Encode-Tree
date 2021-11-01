#pragma once

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric>

using namespace std;

class CSVRow{
public:
	float operator[](size_t index);
	size_t size() const;
	void readNextRow(istream& str);

private:
	vector<string> m_data;
};

istream& operator>>(istream& str, CSVRow& data);

vector<float> process_data(ifstream& file, int& dim);
