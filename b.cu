#include "cuda_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "nlohmann/json.hpp"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

using namespace std;
using namespace nlohmann;
using namespace thrust;

__device__ void concatString(char* destination, char* source);
void resetString(char* str);

struct Product {
	char title[256];
	double price;
	int qty;
	
	__device__ Product operator()(Product a, Product b) {
		a.price += b.price;
		a.qty += b.qty;
	
		concatString(a.title, b.title);
		
		return a;
	}
};

void readData(string file, host_vector<Product> *products);
void printResults(string fileName, Product* results);

int main() {
	// read data
	host_vector<Product> data(25);
	
	readData("./IFF-7-2_Giedrius_Kristinaitis_L3_dat.json", &data);
	
	// copy host data to device data
	device_vector<Product> d_data = data;

	// create initial data
	Product initial;
	
	resetString(initial.title);
	initial.price = 0;
	initial.qty = 0;
	
	// perform reduce
	Product result = reduce(d_data.begin(), d_data.end(), initial, Product());
	
	// print results
	printResults("./IFF-7-2_Giedrius_Kristinaitis_L3_Rez_B.txt", &result);
	
	return 0;
}

// concats to strings
__device__ void concatString(char* destination, char* source) {
	for (int i = 0; i < 256; i++) {
		if (destination[i] == 0) {
			for (int j = 0; j < 256; j++) {
				if (source[j] == 0 || i + j > 255) {
					break;
				}
				
				destination[i + j] = source[j];
			}
			
			break;
		}
	}
}

// resets a string
void resetString(char* str) {
    for (int i = 0; i < 256; i++) {
        str[i] = 0;
    }
}

// prints results to a file
void printResults(string fileName, Product* results) {
	ofstream file;
	
    file.open(fileName, ios_base::out);
	
    file << setw(70) 
		 << "Results"
		 << endl
         << "--------------------------------------------------------------------------------------------------------------------------------------------"
         << endl
         << setw(100) 
		 << "Title |" 
		 << setw(10) 
		 << "Price |" 
		 << setw(10) 
		 << "Quantity" 
		 << endl
         << "--------------------------------------------------------------------------------------------------------------------------------------------"
         << endl;
		 
    string str(results[0].title);
		
    file << setw(100) 
		 << str
		 << " |"
		 << setw(10) 
		 << to_string(results[0].price)
		 << " |"
         << setw(10)
         << to_string(results[0].qty) 
		 << endl;

    file << endl << endl << endl;
}

// reads product data from a file
void readData(string file, host_vector<Product> *products) {
	ifstream input;
    input.open(file, ifstream::in);

    json data = json::parse(input);

    for (auto& element: data["products"]) {
        Product product;
		
		strcpy(product.title, element.value("title", "").c_str());
		
		product.price = element.value("price", 0);
		product.qty = element.value("quantity", 0);

        products->push_back(product);
    }

    input.close();
}

