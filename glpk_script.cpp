// TO DO 
// Make a single script file to get the final order of set B.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
using namespace std;

int n0, n1, m, N;

vector<vector<int>> get_crossing_mat(vector<vector<int>> &graph) {
    vector<vector<int>> C(n1 + 1, vector<int>(n1 + 1, 0));
    for (int i = 1; i <= n1; i++) {
        for (int j = 1; j <= n1; j++) {
            vector<int> Nvi = graph[i + n0];
            vector<int> Nvj = graph[j + n0];

            int crossing = 0;
            int l = 0, r = 0;

            while (l < Nvi.size()) {
                while (r < Nvj.size() && Nvi[l] > Nvj[r]) {
                    r++;
                }
                crossing += r;
                l++;
            }

            C[i][j] = crossing;
        }
    }
    return C;
}

vector<string> simple_tokenizer(string s)
{
    stringstream ss(s);
    string word;
    vector<string> S;
    while (ss >> word) {
        S.push_back(word);
    }
    return S;
}

void readFile(int &a, int &b, int &e, vector<int> &edges_A, vector<int> &edges_B){
    ifstream myfile;
    myfile.open("input.gr");
    
    if(myfile.is_open()){
        while(myfile){
            string myline;
            getline(myfile,myline);
            if(myline.empty()){
                myfile.close();
                return;
            }
            //cout<<myline<<"\n";
            if(myline[0] == 'c')
                continue;
            else if(myline[0] == 'p'){
                vector<string> temp = simple_tokenizer(myline);
                a = stoi(temp[2]);
                b = stoi(temp[3]);
                e = stoi(temp[4]);
                //cout<<a<<b<<e;
            }
            else{
                vector<string> temp = simple_tokenizer(myline);
                //cout<<temp[0]<<" "<<temp[1]<<"\n";
                edges_A.push_back(stoi(temp[0]));
                edges_B.push_back(stoi(temp[1]));
            }
        }
    }
    myfile.close();
}

void solve(vector <int> &edges_A, vector <int> &edges_B) {
    vector<vector<int>> graph(N + 1);
    vector<int> deg(N + 1);

    for (int i = 0; i < m; i++) {
        graph[edges_A[i]].push_back(edges_B[i]);
        graph[edges_B[i]].push_back(edges_A[i]);
    }

    for (int i = 1; i <= N; i++) {
        sort(graph[i].begin(), graph[i].end());
    }

    vector<vector<int>> C = get_crossing_mat(graph);

    ofstream outputFile("glpk_model.mod");

    outputFile << "/* model section */" << endl;
    outputFile << "param n;" <<endl;
    outputFile << "param n0;" <<endl;
    outputFile << "set row;" << endl;
    outputFile << "set col;" << endl;
    outputFile << "param c{row, col};" << endl;
    outputFile << "var m{row, col} binary;" << endl;
    outputFile << "minimize obj: sum{i in row, j in col} m[i,j] * c[i,j];" << endl;
    outputFile << "s.t. transitive{i in row, j in row, k in row : (i != j && j != k && k != i)}:" << endl;
    outputFile << "\tm[i,j] + m[j,k] - m[i,k] <= 1;" << endl;
    outputFile << "s.t. mutual_exclusion{i in row, j in row : (i != j)}:" << endl;
    outputFile << "\tm[i,j] + m[j,i] = 1;" << endl;
    outputFile << "solve;" << endl;
    outputFile << "printf \"Objective value: \%d\\n\", obj;" << endl;
    outputFile << "printf \"\%d \%d\\n\", n, n0 > \"order_matrix.txt\";" << endl;
    outputFile << "for {i in row} {" << endl;
    outputFile << "\tfor {j in col} {" << endl;
    outputFile << "\t \tprintf \"\%d \", m[i,j] >> \"order_matrix.txt\";" << endl;
    outputFile << "\t}" << endl;
    outputFile << "\tprintf \"\\n\" >> \"order_matrix.txt\";" << endl;
    outputFile << "}" << endl;
    outputFile << "/* data section */" << endl;
    outputFile << "data;" << endl;
    outputFile << "param n := "<<n1<<";" << endl;
    outputFile << "param n0 := "<<n0<<";" << endl;
    outputFile << "set row := ";
    for(int i=1; i<=n1; i++){
        outputFile << i << " ";
        if(i == n1)     outputFile << ";"<< endl;
    }
    outputFile << "set col := ";
    for(int i=1; i<=n1; i++){
        outputFile << i << " ";
        if(i == n1)     outputFile << ";"<< endl;
    }
    outputFile << "param c : ";
    for(int i=1; i<=n1; i++){
        outputFile << i << " ";
        if(i == n1)     outputFile << " :="<< endl;
    }
    for(int i=1; i<=n1; i++){
        outputFile << "\t" << i << "\t";
        for(int j=1; j<=n1; j++){
            outputFile << C[i][j] << " ";
        }
        if(i == n1){
            outputFile << ";";
        }
        outputFile << endl;
    }

    outputFile << "end;" << endl;

    outputFile.close();
}

int main() {
    int elementsinA,elementsinB,edges;    
    vector<int> edges_A;
    vector<int> edges_B;
    readFile(elementsinA,elementsinB,edges,edges_A,edges_B);
    n0 = elementsinA;
    n1 = elementsinB;
    m = edges;
    N = n0 + n1;
    solve(edges_A, edges_B);
    return 0;
}