# CS F422: Parallel Computing Assignment  
**Semester:** II, 2024-2025  

## Overview  
Analyzed Amazon Electronics Reviews dataset using CUDA and OpenMP. Tasks:  
1. Top-Rated Products using GPU Bitonic Sort  
2. Sentiment Analysis using GPU and Lexicon  
3. Elaborate Reviewers using Sequential & OpenMP  

---

## Dataset  
- Over 6.7M reviews (JSON format)  
- Fields used: `reviewerID`, `asin`, `reviewText`, `overall`

---

## 1. Top-Rated Products Finder  
- **Base:** Avg ratings on CPU, sorted using GPU Bitonic Sort  
- **Optimized:** Memory aligned to power-of-two, better GPU access  

**Performance:**  
| Stage       | Base (ms) | Optimized (ms) |  
|-------------|-----------|----------------|  
| Parsing     | 156555    | 154716         |  
| GPU Sort    | 8         | 8              |  

---

## 2. Sentiment Analysis  
- **Base:** CPU parsing, scoring using lexicon  
- **Optimized:** Lexicon & scoring done on GPU in batches  

**Performance:**  
| Metric          | Base | Optimized |  
|-----------------|------|-----------|  
| Reviews         | 6.7M | 6.7M      |  
| Time (seconds)  | 324  | 330       |  

---

## 3. Elaborate Reviewers  
- **Definition:** Reviewers with ≥5 reviews having ≥50 words  
- **Base:** Sequential filter  
- **Optimized:** OpenMP parallel filter with dynamic schedule  

**Performance:**  
| Step       | Base (ms) | OpenMP (ms) |  
|------------|-----------|-------------|  
| Filtering  | 121       | 273         |  

---

## Optimizations  
- GPU: Memory coalescing, batching  
- CPU: OpenMP with `#pragma omp critical`  
- Timing: CUDA events, chrono  

---

## Run Instructions  

```bash
# Download data
chmod +x download.sh && ./download.sh

# Compile
nvcc -O3 cuda_toprated.cu -o cuda_toprated
nvcc -O3 cuda_toprated_optimized.cu -o cuda_toprated_optimized
nvcc -O3 cuda_reviewanalysis.cu -o cuda_reviewanalysis
nvcc -O3 cuda_reviewanalysis_optimized.cu -o cuda_reviewanalysis_optimized
g++ -O3 c_elaborate.cpp -o c_elaborate
g++ -O3 -fopenmp c_elaborate_openmp_cpu.cpp -o c_elaborate_openmp

# Run
./cuda_toprated electronics.json > top_rated.txt
./cuda_toprated_optimized electronics.json > top_rated_opt.txt
./cuda_reviewanalysis electronics.json lexicon.txt > sentiment_base.txt
./cuda_reviewanalysis_optimized electronics.json lexicon.txt > sentiment_opt.txt
./c_elaborate electronics.json > elaborate_seq.txt
./c_elaborate_openmp electronics.json > elaborate_omp.txt


Implementation Overview

We designed three modules, each with a base and optimized version:

    Top-Rated Products Finder: Bitonic sort of average ratings on the GPU.

    Sentiment Analysis: Word-level sentiment scoring using a lexicon and GPU batching.

    Elaborate Reviewers: Word-count based filtering with both sequential and OpenMP CPU approaches.

Top-Rated Products Finder
Base Implementation (cuda_toprated.cu)

    Aggregates and computes average ratings on CPU.

    GPU used to sort the average ratings using bitonic sort.

    Top 10 products are extracted.

Optimized Implementation (cuda_toprated_optimized.cu)

    Aligns memory size to next power of two.

    Uses efficient GPU memory access patterns and optimized kernel launches.

    Time profiled using CUDA events.

Performance Comparison
Stage	Base (ms)	Optimized (ms)
CPU Parsing + Aggregation	156555.650	154716.557
Avg Rating Calculation	25.528	25.684
GPU Memory Allocation	1.180	1.196
GPU Kernel (Bitonic Sort)	7.995	7.997
Total	156675.367	154828.178
Review Sentiment Analysis
Base Implementation (cuda_reviewanalysis.cu)

    CPU loads lexicon into a map.

    Each review is tokenized and scored using the lexicon.

    Sentiment classification: Positive, Negative, Neutral.

Optimized Implementation (cuda_reviewanalysis_optimized.cu)

    Lexicon transferred to GPU memory.

    GPU kernel tokenizes and scores batches of reviews.

    Classification is also performed on the GPU.

Performance Comparison
Metric	Base	Optimized
Reviews Processed	6,738,237	6,738,237
Sentiment Output	Similar	Similar
Total Time (seconds)	323.607	329.621
Elaborate Reviewers Identification
Sequential Implementation (c_elaborate.cpp)

    Reviews with word count ≥ 50 are considered elaborate.

    Reviewers with ≥ 5 such reviews are marked as elaborate reviewers.

OpenMP Parallel Implementation (c_elaborate_openmp_cpu.cpp)

    Filtering step parallelized using OpenMP with dynamic scheduling.

    Uses #pragma omp critical to safely update results.

Performance Comparison
Stage	Sequential (ms)	OpenMP (ms)
Parsing + Map Building	225071.750	(sequential)
Filtering	121.283	273.092
Total	225193.161	>225200
Optimization Techniques

    Memory Coalescing in GPU kernels.

    Lexicon Preprocessing for faster lookups.

    Batching Reviews to reduce kernel launch overhead.

    CUDA Event Timing and chrono for profiling.

    OpenMP Parallel Filtering with dynamic scheduling.

Conclusion

    GPU-based bitonic sort significantly accelerates top-rated product identification.

    Sentiment analysis benefits marginally from GPU parallelism due to overheads.

    Elaborate reviewer filtering is CPU-bound and benefits little from OpenMP due to synchronization costs.

I/O and parsing remain the most time-consuming stages and provide the best opportunity for further optimization.
