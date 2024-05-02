#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>

using namespace std;

int main() {
    ifstream inputFile("order_matrix.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening file." << endl;
        return 1;
    }

    int dimension = 0;
    int n0 = 0;

    // Read the first line to get the dimension and n0
    string firstLine;
    if (getline(inputFile, firstLine)) {
        istringstream iss(firstLine);
        if (!(iss >> dimension >> n0)) {
            cerr << "Error reading the matrix dimension and n0." << endl;
            return 1;
        }
    } else {
        cerr << "Empty file or unable to read first line." << endl;
        return 1;
    }

    vector<vector<int>> matrix(dimension, vector<int>(dimension));

    // Read the matrix data from the file
    string line;
    for (int i = 0; i < dimension; i++) {
        if (getline(inputFile, line)) {
            istringstream iss(line);
            for (int j = 0; j < dimension; j++) {
                if (!(iss >> matrix[i][j])) {
                    cerr << "Error reading matrix data." << endl;
                    return 1;
                }
            }
        } else {
            cerr << "Unexpected end of file." << endl;
            return 1;
        }
    }

    vector<int> columnSums(dimension, 0);
    for (int j = 0; j < dimension; j++) {
        for (int i = 0; i < dimension; i++) {
            columnSums[j] += matrix[i][j];
        }
    }

    vector<pair<int, int>> sortedSums;
    for (int j = 0; j < dimension; j++) {
        sortedSums.push_back({columnSums[j], j});
    }

    sort(sortedSums.begin(), sortedSums.end());

    for (auto &p : sortedSums) {
        cout << p.second + n0 + 1 << " ";
    }
    cout << endl;

    return 0;
}
