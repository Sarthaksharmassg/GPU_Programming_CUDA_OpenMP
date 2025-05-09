// Full CUDA Bitonic Sort with Shared Memory and Optimizations
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <utility>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <chrono>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Failure in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

using namespace std;

__global__ void performBitonicStageShared(float *values, int *indices, int stage, int pass, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    extern __shared__ char sharedMemory[];
    float *sharedVals = (float *)sharedMemory;
    int *sharedIndices = (int *)&sharedVals[blockDim.x];

    if (idx < count) {
        sharedVals[tid] = values[idx];
        sharedIndices[tid] = indices[idx];
    }

    __syncthreads();

    int pairDistance = 1 << (stage - pass);
    int blockSize = 1 << stage;
    int partner = idx ^ pairDistance;
    bool ascending = ((idx & blockSize) == 0);

    if (partner < count && tid < blockDim.x) {
        int localPartner = partner - blockIdx.x * blockDim.x;
        if (localPartner >= 0 && localPartner < blockDim.x) {
            float val1 = sharedVals[tid];
            float val2 = sharedVals[localPartner];
            int ind1 = sharedIndices[tid];
            int ind2 = sharedIndices[localPartner];

            bool doSwap = (ascending && val1 > val2) || (!ascending && val1 < val2);

            if (doSwap) {
                sharedVals[tid] = val2;
                sharedVals[localPartner] = val1;
                sharedIndices[tid] = ind2;
                sharedIndices[localPartner] = ind1;
            }
        }
    }

    __syncthreads();

    if (idx < count) {
        values[idx] = sharedVals[tid];
        indices[idx] = sharedIndices[tid];
    }
}

int calculateNextPower2(int x) {
    if (x <= 0) return 1;
    int p = 1;
    while (p < x) p *= 2;
    return p;
}

void computeAverages(const map<string, pair<double, int>> &aggregates,
                     vector<float> &ratings,
                     vector<string> &products,
                     vector<int> &original_indices) {
    ratings.reserve(aggregates.size());
    products.reserve(aggregates.size());
    original_indices.reserve(aggregates.size());
    int index = 0;
    for (const auto &entry : aggregates) {
        if (entry.second.second > 0) {
            ratings.push_back(static_cast<float>(entry.second.first / entry.second.second));
            products.push_back(entry.first);
            original_indices.push_back(index++);
        }
    }
}

int main(int argc, char **argv) {
    auto start_time = chrono::high_resolution_clock::now();

    if (argc != 2) {
        cerr << "Provide JSON file path as argument.\n";
        return 1;
    }

    ifstream input(argv[1]);
    if (!input.is_open()) {
        cerr << "Unable to read file: " << argv[1] << endl;
        return 1;
    }

    cout << "Loading and aggregating reviewsd..." << endl;
    map<string, pair<double, int>> review_map;
    string row;
    long long lines = 0, valid_lines = 0, bad_lines = 0;
    rapidjson::Document doc;

    while (getline(input, row)) {
        lines++;
        doc.Parse(row.c_str());
        if (doc.HasParseError()) { bad_lines++; continue; }

        if (doc.IsObject() && doc.HasMember("asin") && doc["asin"].IsString()
            && doc.HasMember("overall") && doc["overall"].IsNumber()) {
            try {
                string id = doc["asin"].GetString();
                float rating = doc["overall"].GetFloat();
                review_map[id].first += rating;
                review_map[id].second++;
                valid_lines++;
            } catch (const exception &e) {
                bad_lines++;
            }
        } else bad_lines++;
    }
    input.close();

    cout << "Reviews loaded: " << valid_lines << " for " << review_map.size() << " items." << endl;
    if (bad_lines > 0) cout << "Ignored " << bad_lines << " malformed entries." << endl;
    if (review_map.empty()) return 1;

    vector<float> averages;
    vector<string> asin_list;
    vector<int> indices;
    computeAverages(review_map, averages, asin_list, indices);
    int total_items = averages.size();
    if (total_items == 0) return 1;

    cout << "Sorting using GPU..." << endl;

    cudaEvent_t t_start, t_end;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_end));
    float h2d_time = 0, exec_time = 0, d2h_time = 0;

    int padded = calculateNextPower2(total_items);
    vector<float> padded_values(padded, -1.0f);
    vector<int> padded_ids(padded, -1);
    copy(averages.begin(), averages.end(), padded_values.begin());
    copy(indices.begin(), indices.end(), padded_ids.begin());

    float *d_vals;
    int *d_inds;
    CUDA_CHECK(cudaEventRecord(t_start));
    CUDA_CHECK(cudaMalloc(&d_vals, padded * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inds, padded * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_vals, padded_values.data(), padded * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inds, padded_ids.data(), padded * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t_end));
    CUDA_CHECK(cudaEventSynchronize(t_end));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_time, t_start, t_end));

    int steps = static_cast<int>(log2(padded));
    int threads = 256;
    int blocks = (padded + threads - 1) / threads;
    int sharedMemSize = threads * (sizeof(float) + sizeof(int));

    CUDA_CHECK(cudaEventRecord(t_start));
    for (int stage = 0; stage < steps; ++stage) {
        for (int pass = 0; pass <= stage; ++pass) {
            performBitonicStageShared<<<blocks, threads, sharedMemSize>>>(d_vals, d_inds, stage + 1, pass + 1, padded);
        }
    }
    CUDA_CHECK(cudaEventRecord(t_end));
    CUDA_CHECK(cudaEventSynchronize(t_end));
    CUDA_CHECK(cudaEventElapsedTime(&exec_time, t_start, t_end));

    vector<int> sorted_indices(padded);
    CUDA_CHECK(cudaEventRecord(t_start));
    CUDA_CHECK(cudaMemcpy(sorted_indices.data(), d_inds, padded * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_inds));
    CUDA_CHECK(cudaEventRecord(t_end));
    CUDA_CHECK(cudaEventSynchronize(t_end));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time, t_start, t_end));

    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_end));

    cout << "GPU sort complete.\n\n--- Top 10 Products ---\n";
    int top_n = min(total_items, 10), shown = 0;
    cout << fixed << setprecision(4);
    for (int i = 0; i < padded && shown < top_n; ++i) {
        int idx = sorted_indices[i];
        if (idx >= 0 && idx < total_items) {
            cout << shown + 1 << ". Product ID: " << asin_list[idx]
                 << " | Rating: " << averages[idx] << endl;
            shown++;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> full_duration = end_time - start_time;

    cout << "\n--- Performance Stats ---\n";
    cout << fixed << setprecision(3);
    cout << "Memory Transfer to GPU:       " << h2d_time << " ms\n";
    cout << "GPU Bitonic Sort Time:        " << exec_time << " ms\n";
    cout << "Transfer from GPU & Cleanup:  " << d2h_time << " ms\n";
    cout << "Total Runtime:                " << full_duration.count() << " ms\n";

    return 0;
}
