#include<iostream>
#include<vector>
#include<fstream>
#include <sstream>
#include<ctime>
using namespace std;
vector<int> answer;

//TODO
//1. Add condition for complete graph
void swap(int& x, int& y) 
{ 
    int temp = x; 
    x = y; 
    y = temp; 
} 
int numberOfCuts(vector<int>& edges_A, vector<int>& edges_B, int edges, unordered_map<int,int> &umap)
{
    int cuts = 0;
    for(int i = 0 ; i<edges ; i++)
    {
        for(int j = i+1 ; j <edges ; j++) //another alternative for would be sort edges based on A, and iterate from j = i+1. NOT REQUIRED. EDGES ARE SORTED
        {
            // cout<< "nums " << A[i] << " " << A[j] << " " << B[i] << " "<< B[j]<<"\n";
            // cout<<"index" << umap[A[i]] << " " << umap[A[j]]<< " " << umap[B[i]] << " " << umap[B[j]]<< "\n";
            if(umap[edges_A[i]] < umap[edges_A[j]] && umap[edges_B[i]] > umap[edges_B[j]])
            {
                cuts++;
            }
                
        }
    }
    return cuts;
}

int numberOfCutsUsingSegmentTree(vector<int>& edges_A, vector<int>& edges_B, int edges, unordered_map<int,int> &umap)
{

}

void generateAllPermutationsRecursive(vector<int> &edges_A, vector<int> &edges_B, vector<int> &B, int &minCuts, int l, int h,  unordered_map<int,int> &umap, int &edges)
{
    if(l == h)
    {
        int temp = numberOfCuts(edges_A, edges_B, edges,umap) ;
        if(minCuts > temp)
        {
            minCuts = temp;
            answer = B;
            return;
        }
        return;
        // return numberOfCuts(edges_A, edges_B, edges,umap);
       // return 1;
    }
    // int count = 0;
    // int minCuts = 1e9;
    for(int i=l ; i<=h ; i++)
    {
        swap(B[l],B[i]);
        swap(umap[B[l]],umap[B[i]]);
        
        //count += generateAllPermutationsRecursive(A, B,l+1,h, minCuts, umap, edges);
        generateAllPermutationsRecursive(edges_A, edges_B,B,minCuts,l+1,h, umap, edges);
        // if(temp < minCuts)
        // {
        //     minCuts = temp;
        //     answer = B;
        //     cout<<"\n";
        //     for(int i=0 ; i<B.size() ; i++)
        //         cout<<B[i]<<" ";
        //     cout<<"\n";
        // }

        swap(B[l],B[i]);
        swap(umap[B[l]],umap[B[i]]);
    }  
    // return count;
    // return minCuts;
    
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

void readFile(int &a, int &b, int &e, vector<int> &edges_A, vector<int> &edges_B)
{
    ifstream myfile;
    myfile.open("/Users/spatel2/ResearchStuff/PACE/tiny_test_set/complete_4_5.gr");
    
    if(myfile.is_open())
    {
        while(myfile)
        {
            string myline;
            getline(myfile,myline);
            if(myline.empty())
            {
                myfile.close();
                return;
            }
            //cout<<myline<<"\n";
            if(myline[0] == 'c')
                continue;
            else if(myline[0] == 'p')
            {
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
int main()
{    
    int elementsinA,elementsinB,edges;    
    vector<int> edges_A;
    vector<int> edges_B;
    readFile(elementsinA,elementsinB,edges,edges_A,edges_B);

    int total = elementsinA + elementsinB;
    vector<int> B(elementsinB);
    for(int i=0 ; i<elementsinB; i++)
    {
        B[i] = i+elementsinA+1;
    }

    unordered_map<int,int> numToIndex;
    for(int i=1 ; i<=total ; i++)
    {
        numToIndex[i] = i;
    } 

    answer = B;

    int startTime = time(NULL);


   int minCuts = numberOfCuts(edges_A,edges_B,edges, numToIndex);

   if(minCuts == 0)
   {
        cout<<"No  further computation needed, mincuts : 0\n";
        return 0;
   }


    //condition for complete bipartite graph
    if(edges_A.size() == elementsinA * elementsinB)
    {
        cout<<"No  further computation needed, mincuts : "<<elementsinA * (elementsinA - 1) * elementsinB * (elementsinB - 1) / 4<<"\n";
        return elementsinA * (elementsinA - 1) * elementsinB * (elementsinB - 1) / 4;
    }
        
   // cout<<"\n"<<generateAllPermutationsRecursive(edges_A, edges_B, B, 0,elementsinB-1, minCuts, numToIndex, edges, answer)<<"\n";
   // cout<<"\n"<<generateAllPermutationsRecursive(edges_A, edges_B, B, minCuts, 0,elementsinB-1, numToIndex, edges)<<"\n";
    generateAllPermutationsRecursive(edges_A, edges_B, B, minCuts, 0,elementsinB-1, numToIndex, edges);

    int endTime = time(NULL);

    cout<<"Time Taken : "<<endTime-startTime<<"\n";
    
    cout<<"\n"<<minCuts<<"\n";
    // for(int i =0 ; i<elementsinB ; i++)
    //     cout<<answer[i]<<" ";
    // cout<<"\n";

}