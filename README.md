# GPU_Programming_CUDA_OpenMP
GPU progg Data Analysis on Data Scraped from Amzn (intentional abrev) using CUDA and OpenMP 
Steps:

1.Run the download.sh to download the dataset and lexicon file.
2.Compile the files with nvcc and then ececute them individually no makefile has been provided yet
3.For OpemMP GPU offloading supports are required which are provided with newer versions of compilers like gcc not in the older versions however clang provides it as inbuilt support.In this code the OpenMP code essentially works on CPU due the before mentioned limitation in our admin controlled cluster config.
