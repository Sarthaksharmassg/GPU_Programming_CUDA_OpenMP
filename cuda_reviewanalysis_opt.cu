#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <iomanip>
#include <cctype>
#include <algorithm>
#include <chrono>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include <cuda_runtime.h>

using namespace std;

#define VERIFY_CUDA(err)                                            \
    if (err != cudaSuccess)                                         \
    {                                                               \
        fprintf(stderr, "CUDA failure at %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE);                                         \
    }

string formatText(const string &s)
{
    size_t start = s.find_first_not_of(" \t\n\r\f\v.,!?;:\"'()[]{}");
    if (start == string::npos) return s;
    size_t end = s.find_last_not_of(" \t\n\r\f\v.,!?;:\"'()[]{}");
    return s.substr(start, end - start + 1);
}

string makeLower(string s)
{
    transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return tolower(c); });
    return s;
}

__global__ void analyzeSentiment(const int *words, const int *offsets,
                                 const int *lengths, const float *scores,
                                 int total, float *results, int scoreCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int off = offsets[idx];
    int len = lengths[idx];
    float sum = 0.0f;

    for (int i = 0; i < len; ++i)
    {
        int word = words[off + i];
        if (word >= 0 && word < scoreCount)
        {
            sum += scores[word];
        }
    }

    results[idx] = sum;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "Provide paths: " << argv[0] << " <reviews.json> <lexicon.txt>" << endl;
        return 1;
    }

    string reviewsPath = argv[1];
    string lexiconPath = argv[2];

    auto begin = chrono::high_resolution_clock::now();

    cout << "Opening lexicon from: " << lexiconPath << endl;
    unordered_map<string, int> wordMap;
    vector<float> cpuScores;
    ifstream lexicon(lexiconPath);

    if (!lexicon.is_open())
    {
        cerr << "Could not open lexicon: " << lexiconPath << endl;
        return 1;
    }

    string entry;
    int index = 0;
    while (getline(lexicon, entry))
    {
        stringstream ss(entry);
        string term;
        float val;
        if (getline(ss, term, '\t') && ss >> val)
        {
            string clean = makeLower(formatText(term));
            if (!clean.empty() && wordMap.find(clean) == wordMap.end())
            {
                wordMap[clean] = index++;
                cpuScores.push_back(val);
            }
        }
    }
    lexicon.close();

    int scoreSize = cpuScores.size();
    if (scoreSize == 0)
    {
        cerr << "No entries in lexicon." << endl;
        return 1;
    }

    cout << "Identified " << scoreSize << " lexicon terms." << endl;

    float *gpuScores = nullptr;
    size_t scoreBytes = scoreSize * sizeof(float);
    cout << "GPU memory requested: " << scoreBytes / (1024.0 * 1024.0) << " MB" << endl;
    VERIFY_CUDA(cudaMalloc(&gpuScores, scoreBytes));
    VERIFY_CUDA(cudaMemcpy(gpuScores, cpuScores.data(), scoreBytes, cudaMemcpyHostToDevice));

    cout << "Reading reviews from: " << reviewsPath << endl;
    ifstream reviews(reviewsPath);
    if (!reviews.is_open())
    {
        cerr << "Could not open reviews: " << reviewsPath << endl;
        VERIFY_CUDA(cudaFree(gpuScores));
        return 1;
    }

    const size_t CHUNK = 262144;
    vector<string> texts;
    texts.reserve(CHUNK);
    rapidjson::Document doc;

    long long lines = 0, good = 0, bad = 0, pos = 0, neg = 0, neu = 0;
    string buffer;

    while (true)
    {
        texts.clear();
        while (texts.size() < CHUNK && getline(reviews, buffer))
        {
            lines++;
            doc.Parse(buffer.c_str());

            if (doc.HasParseError()) { bad++; continue; }

            if (doc.IsObject() && doc.HasMember("reviewText") && doc["reviewText"].IsString())
                texts.push_back(doc["reviewText"].GetString());
            else
                bad++;
        }

        if (texts.empty()) break;

        size_t batch = texts.size();
        vector<int> all_ids, offs, lens;
        offs.reserve(batch);
        lens.reserve(batch);
        int offset = 0;

        for (const string &r : texts)
        {
            offs.push_back(offset);
            int count = 0;
            stringstream ss(r);
            string w;
            while (ss >> w)
            {
                string clean = makeLower(formatText(w));
                if (!clean.empty())
                {
                    auto it = wordMap.find(clean);
                    all_ids.push_back(it != wordMap.end() ? it->second : -1);
                    count++;
                }
            }
            lens.push_back(count);
            offset += count;
        }

        int *ids_gpu = nullptr, *offs_gpu = nullptr, *lens_gpu = nullptr;
        float *res_gpu = nullptr;

        VERIFY_CUDA(cudaMalloc(&ids_gpu, all_ids.size() * sizeof(int)));
        VERIFY_CUDA(cudaMalloc(&offs_gpu, offs.size() * sizeof(int)));
        VERIFY_CUDA(cudaMalloc(&lens_gpu, lens.size() * sizeof(int)));
        VERIFY_CUDA(cudaMalloc(&res_gpu, batch * sizeof(float)));

        VERIFY_CUDA(cudaMemcpy(ids_gpu, all_ids.data(), all_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        VERIFY_CUDA(cudaMemcpy(offs_gpu, offs.data(), offs.size() * sizeof(int), cudaMemcpyHostToDevice));
        VERIFY_CUDA(cudaMemcpy(lens_gpu, lens.data(), lens.size() * sizeof(int), cudaMemcpyHostToDevice));

        int tpb = 256;
        int bpg = (batch + tpb - 1) / tpb;

        analyzeSentiment<<<bpg, tpb>>>(ids_gpu, offs_gpu, lens_gpu, gpuScores, batch, res_gpu, scoreSize);
        VERIFY_CUDA(cudaGetLastError());

        vector<float> hostScores(batch);
        VERIFY_CUDA(cudaMemcpy(hostScores.data(), res_gpu, batch * sizeof(float), cudaMemcpyDeviceToHost));

        VERIFY_CUDA(cudaFree(ids_gpu));
        VERIFY_CUDA(cudaFree(offs_gpu));
        VERIFY_CUDA(cudaFree(lens_gpu));
        VERIFY_CUDA(cudaFree(res_gpu));

        for (float s : hostScores)
        {
            good++;
            if (s > 0) pos++;
            else if (s < 0) neg++;
            else neu++;
        }

        cout << "Completed lines up to: " << lines << " - Processed: " << batch << endl;

        if (reviews.eof() && texts.size() < CHUNK) break;
    }

    reviews.close();
    VERIFY_CUDA(cudaFree(gpuScores));

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - begin;

    cout << "\n=== Analysis Overview ===" << endl;
    cout << "Lines scanned:        " << lines << endl;
    cout << "Entries analyzed:     " << good << endl;
    cout << "Failed reads:         " << bad << endl;
    cout << "Positive entries:     " << pos << endl;
    cout << "Negative entries:     " << neg << endl;
    cout << "Neutral entries:      " << neu << endl;
    cout << "Duration Checking:          " << fixed << setprecision(3) << duration.count() << " sec" << endl;

    return 0;
}
