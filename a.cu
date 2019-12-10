#include "cuda_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "nlohmann/json.hpp"
#include "device_launch_parameters.h"

using namespace std;
using namespace nlohmann;

struct Product {
	char title[256];
	double price;
	int qty;
};

void readData(string file, vector<Product> *products);
void printResults(string fileName, Product* results, int resultCount);
__global__ void sum(Product* products, Product* results, int* rangeLength, int* productCount);
__device__ void concatString(char* destination, char* source);
__device__ void resetString(char* str);

// entry point of the program
int main() {
	vector<Product> products;
	
	readData("./IFF-7-2_Giedrius_Kristinaitis_L3_dat.json", &products);
	
	// host data
	Product* productPtr = &products[0];
	int productCount = products.size();
	int threadCount = 8;
	int rangeLength = products.size() / threadCount;
	Product* resultsPtr = new Product[threadCount];
	
	// device data
	Product* d_products;
	Product* d_results;
	int* d_rangeLength;
	int* d_productCount;
	
	// allocate device memory
	cudaMalloc((void**) &d_products, productCount * sizeof(Product));
	cudaMalloc((void**) &d_results, threadCount * sizeof(Product));
	cudaMalloc((void**) &d_rangeLength, sizeof(int));
	cudaMalloc((void**) &d_productCount, sizeof(int));
	
	// copy host data to device memory
	cudaMemcpy(d_products, productPtr, productCount * sizeof(Product), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rangeLength, &rangeLength, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_productCount, &productCount, sizeof(int), cudaMemcpyHostToDevice);
	
	// perform calculations
	sum<<<1, threadCount>>>(d_products, d_results, d_rangeLength, d_productCount);

	// wait for other threads to finish
	cudaDeviceSynchronize();
	
	// copy results to host
	cudaMemcpy(resultsPtr, d_results, threadCount * sizeof(Product), cudaMemcpyDeviceToHost);
	
	// print results
	printResults("./IFF-7-2_Giedrius_Kristinaitis_L3_Rez_A.txt", resultsPtr, threadCount);
	
	// free memory
	cudaFree(d_products);
	cudaFree(d_results);
	cudaFree(d_rangeLength);
	cudaFree(d_productCount);
	free(resultsPtr);
	
	return 0;
}

// performs calculations
__global__ void sum(Product* products, Product* results, int* rangeLength, int* productCount) {
	int startIndex = threadIdx.x * *rangeLength;
    int endIndex = min(startIndex + *rangeLength, *productCount);

	Product result;
	
	resetString(result.title);
	result.price = 0;
	result.qty = 0;
	
	for (int i = startIndex; i < endIndex; i++) {
		Product current = products[i];
		
		result.price += current.price;
		result.qty += current.qty;
		concatString(result.title, current.title);
	}
	
	results[threadIdx.x] = result;
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
__device__ void resetString(char* str) {
    for (int i = 0; i < 256; i++) {
        str[i] = 0;
    }
}

// prints results to a file
void printResults(string fileName, Product* results, int resultCount) {
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
		 
    for (int i = 0; i < resultCount; ++i) {
        string str(results[i].title);
		
        file << setw(100) 
			 << str
			 << " |"
			 << setw(10) 
			 << to_string(results[i].price)
			 << " |"
             << setw(10)
             << to_string(results[i].qty) 
			 << endl;
    }

    file << endl << endl << endl;
}

// reads product data from a file
void readData(string file, vector<Product> *products) {
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