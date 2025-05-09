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

#define VERIFY_CUDA(err)                                      \
    if (err != cudaSuccess)                                   \
    {                                                         \
        fprintf(stderr, "CUDA failed in %s:%d: %s\n",         \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }

using namespace std;

string refineText(const string &s) {
    size_t begin = s.find_first_not_of(" \t\n\r\f\v.,!?;:\"'()[]{}");
    //empty basically null string
    if (begin == string::npos) return s;
    size_t end = s.find_last_not_of(" \t\n\r\f\v.,!?;:\"'()[]{}");
    //because begins has changed
    return s.substr(begin, end - begin + 1);
}

string normalizeText(string str) {
    //case transfrm 
    transform(str.begin(), str.end(), str.begin(),
              [](unsigned char ch) { return tolower(ch); });
    return str;
}

int main(int argc, char **argv) {
    //exec json lexicon rules 
    if (argc != 3) {
        cerr << "Provide the correct number of arguments!" << endl;
        return 1;
    }
    //json
    string reviewPath = argv[1];
    //lexicon score txt file
    string wordScorePath = argv[2];

    auto timeStart = chrono::high_resolution_clock::now();

    cout << "Reading sentiment words from: " << wordScorePath << endl;
    //string and correspondin score
    unordered_map<string, float> sentimentWords;
    //creates input filestream for the lexicon.txt
    ifstream wordFile(wordScorePath);
    if (!wordFile.is_open()) {
        cerr << "Couldn't load word list from: " << wordScorePath << endl;
        return 1;
    }

    string fileLine;
    int wordCount = 0;
    while (getline(wordFile, fileLine)) {
        stringstream stream(fileLine);
        string term;
        float val;
        if (getline(stream, term, '\t') && (stream >> val)) {
            sentimentWords[term] = val;
            wordCount++;
        }
    }
    wordFile.close();
    cout << "Terms captured: " << wordCount << endl;

    if (sentimentWords.empty()) {
        cerr << "Empty or unreadable word list." << endl;
        return 1;
    }

    cout << "Analyzing reviews from: " << reviewPath << endl;
    ifstream reviewFile(reviewPath);
    if (!reviewFile.is_open()) {
        cerr << "Couldn't open review file: " << reviewPath << endl;
        return 1;
    }

    long long linesTotal = 0, linesUsed = 0, linesErrored = 0;
    long long positive = 0, negative = 0, neutral = 0;

    rapidjson::Document doc;

    int cudaDevices;
    VERIFY_CUDA(cudaGetDeviceCount(&cudaDevices));
    if (cudaDevices > 0) {
        cout << "CUDA devices available: " << cudaDevices << endl;
        VERIFY_CUDA(cudaSetDevice(0));
    } else {
        cout << "No CUDA devices found." << endl;
    }

    while (getline(reviewFile, fileLine)) {
        linesTotal++;
        doc.Parse(fileLine.c_str());

        if (doc.HasParseError()) {
            linesErrored++;
            continue;
        }

        if (doc.IsObject() && doc.HasMember("reviewText") && doc["reviewText"].IsString()) {
            try {
                const char *textPtr = doc["reviewText"].GetString();
                string review(textPtr);

                float totalScore = 0.0f;
                int matchedWords = 0;
                stringstream wordsStream(review);
                string word;

                while (wordsStream >> word) {
                    string cleaned = normalizeText(refineText(word));
                    if (!cleaned.empty()) {
                        auto search = sentimentWords.find(cleaned);
                        if (search != sentimentWords.end()) {
                            totalScore += search->second;
                            matchedWords++;
                        }
                    }
                }

                string sentiment = "neutral";
                if (totalScore > 0) {
                    sentiment = "positive";
                    positive++;
                } else if (totalScore < 0) {
                    sentiment = "negative";
                    negative++;
                } else {
                    neutral++;
                }

                linesUsed++;
            }
            catch (const exception &ex) {
                cerr << "Issue occurred on line " << linesTotal << ": " << ex.what() << endl;
                linesErrored++;
            }
        } else {
            linesErrored++;
        }

        if (linesTotal % 100000 == 0) {
            cout << "Lines processed: " << linesTotal << endl;
        }
    }
    reviewFile.close();

    auto timeEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> totalTime = timeEnd - timeStart;

    cout << "\n=== Summary Report ===" << endl;
    cout << "Total lines parsed:      " << linesTotal << endl;
    cout << "Lines evaluated:         " << linesUsed << endl;
    cout << "Invalid entries:         " << linesErrored << endl;
    cout << "-------------------------------" << endl;
    cout << "Positive: " << positive << endl;
    cout << "Negative: " << negative << endl;
    cout << "Neutral : " << neutral << endl;
    cout << "-------------------------------" << endl;
    cout << "Time taken: " << fixed << setprecision(3) << totalTime.count() << " seconds" << endl;
    cout << "-------------------------------" << endl;

    VERIFY_CUDA(cudaDeviceSynchronize());
    cout << "CUDA sync complete." << endl;

    return 0;
}
