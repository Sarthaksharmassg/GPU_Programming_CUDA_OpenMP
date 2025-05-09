#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>

#define MAX_LINE_LENGTH 10000
#define MAX_REVIEWERS 100000
#define MAX_ID_LENGTH 20
#define CHUNK_SIZE 1000  

typedef struct {
    char reviewerID[MAX_ID_LENGTH];
    char reviewerName[100];
    int elaborateReviewCount;
    omp_lock_t lock;  
} Reviewer;

typedef struct {
    char reviewerID[MAX_ID_LENGTH];
    char reviewerName[100];
    int wordCount;
    bool isElaborate; 
} ReviewData;

int countWords(const char *text) {
    int count = 0;
    bool inWord = false;
    
    while (*text) {
        if (isspace(*text)) {
            inWord = false;
        } else if (!inWord) {
            inWord = true;
            count++;
        }
        text++;
    }
    
    return count;
}

void extractStringBetweenQuotes(const char *source, char *target, int max_size) {
    const char *start = strchr(source, '"');
    if (start == NULL) {
        target[0] = '\0';
        return;
    }
    start++;
    const char *end = strchr(start, '"');
    if (end == NULL) {
        target[0] = '\0';
        return;
    }
    int length = end - start;
    if (length >= max_size) {
        length = max_size - 1;
    }
    strncpy(target, start, length);
    target[length] = '\0';
}

bool extractField(const char *line, const char *fieldName, char *result, int max_size) {
    char searchStr[100];
    sprintf(searchStr, "\"%s\":", fieldName);
    
    char *fieldStart = strstr(line, searchStr);
    if (fieldStart == NULL) {
        result[0] = '\0';
        return false;
    }
    fieldStart += strlen(searchStr);
    while (isspace(*fieldStart)) {
        fieldStart++;
    }
    
    if (*fieldStart == '"') {
        extractStringBetweenQuotes(fieldStart, result, max_size);
    } else {
        char *endPos = strpbrk(fieldStart, ",}");
        if (endPos == NULL) {
            result[0] = '\0';
            return false;
        }
        
        int length = endPos - fieldStart;
        if (length >= max_size) {
            length = max_size - 1;
        }
        
        strncpy(result, fieldStart, length);
        result[length] = '\0';
    }
    
    return true;
}

int findReviewer(Reviewer *reviewers, int count, const char *reviewerID) {
    for (int i = 0; i < count; i++) {
        if (strcmp(reviewers[i].reviewerID, reviewerID) == 0) {
            return i;
        }
    }
    return -1;
}

void processReviewsBatch(char **lines, int lineCount, ReviewData *results) {
    #pragma omp parallel for
    for (int i = 0; i < lineCount; i++) {
        char reviewerID[MAX_ID_LENGTH];
        char reviewerName[100];
        char reviewText[MAX_LINE_LENGTH];
        if (!extractField(lines[i], "reviewerID", reviewerID, MAX_ID_LENGTH)) {
            results[i].isElaborate = false;
            continue;
        }
        
        extractField(lines[i], "reviewerName", reviewerName, 100);
        
        if (!extractField(lines[i], "reviewText", reviewText, MAX_LINE_LENGTH)) {
            results[i].isElaborate = false;
            continue;
        }
        int wordCount = countWords(reviewText);
        strcpy(results[i].reviewerID, reviewerID);
        strcpy(results[i].reviewerName, reviewerName);
        results[i].wordCount = wordCount;
        results[i].isElaborate = (wordCount >= 50);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    clock_t start_time = clock();
    double start_omp = omp_get_wtime();
    int num_threads = omp_get_max_threads();
    printf("Using up to %d OpenMP threads\n", num_threads);
    
    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", argv[1]);
        return 1;
    }
    long totalLines = 0;
    char countBuffer[MAX_LINE_LENGTH];
    while (fgets(countBuffer, MAX_LINE_LENGTH, file)) {
        totalLines++;
    }
    rewind(file);
    
    printf("Processing %ld reviews...\n", totalLines);
    Reviewer *reviewers = (Reviewer *)malloc(MAX_REVIEWERS * sizeof(Reviewer));
    if (reviewers == NULL) {
        printf("Memory allocation failed for reviewers\n");
        fclose(file);
        return 1;
    }
    
    for (int i = 0; i < MAX_REVIEWERS; i++) {
        reviewers[i].elaborateReviewCount = 0;
        reviewers[i].reviewerID[0] = '\0';
        reviewers[i].reviewerName[0] = '\0';
        omp_init_lock(&reviewers[i].lock);
    }
    
    int reviewerCount = 0;
    omp_lock_t reviewerCountLock;
    omp_init_lock(&reviewerCountLock);
    
    char **lineBuffers = (char **)malloc(CHUNK_SIZE * sizeof(char *));
    if (lineBuffers == NULL) {
        printf("Memory allocation failed for line buffers\n");
        free(reviewers);
        fclose(file);
        return 1;
    }
    
    for (int i = 0; i < CHUNK_SIZE; i++) {
        lineBuffers[i] = (char *)malloc(MAX_LINE_LENGTH * sizeof(char));
        if (lineBuffers[i] == NULL) {
            printf("Memory allocation failed for line buffer %d\n", i);
            for (int j = 0; j < i; j++) {
                free(lineBuffers[j]);
            }
            free(lineBuffers);
            free(reviewers);
            fclose(file);
            return 1;
        }
    }
    ReviewData *results = (ReviewData *)malloc(CHUNK_SIZE * sizeof(ReviewData));
    if (results == NULL) {
        printf("Memory allocation failed for results\n");
        for (int i = 0; i < CHUNK_SIZE; i++) {
            free(lineBuffers[i]);
        }
        free(lineBuffers);
        free(reviewers);
        fclose(file);
        return 1;
    }
    long processedLines = 0;
    long lastReportedPercentage = 0;
    
    while (!feof(file)) {
        int linesRead = 0;
        for (int i = 0; i < CHUNK_SIZE && !feof(file); i++) {
            if (fgets(lineBuffers[i], MAX_LINE_LENGTH, file) != NULL) {
                linesRead++;
            }
        }
        
        if (linesRead == 0) {
            break;
        }
        processReviewsBatch(lineBuffers, linesRead, results);
        #pragma omp parallel for
        for (int i = 0; i < linesRead; i++) {
            if (results[i].isElaborate) {
                int index = -1;
                #pragma omp critical(find_reviewer)
                {
                    index = findReviewer(reviewers, reviewerCount, results[i].reviewerID);
                }
                
                if (index == -1) {
                    omp_set_lock(&reviewerCountLock);
                    if (reviewerCount < MAX_REVIEWERS) {
                        strcpy(reviewers[reviewerCount].reviewerID, results[i].reviewerID);
                        strcpy(reviewers[reviewerCount].reviewerName, results[i].reviewerName);
                        reviewers[reviewerCount].elaborateReviewCount = 1;
                        index = reviewerCount;
                        reviewerCount++;
                    }
                    omp_unset_lock(&reviewerCountLock);
                } else {
                    omp_set_lock(&reviewers[index].lock);
                    reviewers[index].elaborateReviewCount++;
                    omp_unset_lock(&reviewers[index].lock);
                }
            }
        }
        
        processedLines += linesRead;
        long percentage = (processedLines * 100) / totalLines;
        if (percentage > lastReportedPercentage && percentage % 10 == 0) {
            printf("Progress: %ld%%\n", percentage);
            lastReportedPercentage = percentage;
        }
    }
    
    clock_t end_time = clock();
    double end_omp = omp_get_wtime();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    double omp_time_used = end_omp - start_omp;
    
    printf("\nPerformance Metrics:\n");
    printf("CPU Time: %.2f seconds\n", cpu_time_used);
    printf("OpenMP Wall Time: %.2f seconds\n", omp_time_used);
    printf("Processed %.2f reviews per second\n", totalLines / omp_time_used);
    printf("\nReviewers with at least 5 elaborate reviews (>=50 words):\n");
    printf("-------------------------------------------------------\n");
    printf("%-20s %-30s %s\n", "ReviewerID", "Reviewer Name", "Elaborate Review Count");
    printf("-------------------------------------------------------\n");
    
    int foundCount = 0;
    for (int i = 0; i < reviewerCount; i++) {
        if (reviewers[i].elaborateReviewCount >= 5) {
            printf("%-20s %-30s %d\n", reviewers[i].reviewerID, reviewers[i].reviewerName, 
                   reviewers[i].elaborateReviewCount);
            foundCount++;
        }
    }
    
    printf("-------------------------------------------------------\n");
    printf("Total: %d reviewers found\n", foundCount);
    omp_destroy_lock(&reviewerCountLock);
    for (int i = 0; i < MAX_REVIEWERS; i++) {
        omp_destroy_lock(&reviewers[i].lock);
    }
    free(results);
    for (int i = 0; i < CHUNK_SIZE; i++) {
        free(lineBuffers[i]);
    }
    free(lineBuffers);
    free(reviewers);
    fclose(file);
    
    return 0;
}