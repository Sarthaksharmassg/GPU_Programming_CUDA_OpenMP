#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <time.h>
#define MAX_LINE_LENGTH 10000
#define MAX_REVIEWERS 100000
#define MAX_ID_LENGTH 20

typedef struct {
    char reviewerID[MAX_ID_LENGTH];
    char reviewerName[100];
    int elaborateReviewCount;
} Reviewer;

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
        // Non-string value (like numbers, booleans)
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

int main(int argc, char *argv[]) {
    time_t startTime;
    startTime=time(NULL);
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", argv[1]);
        return 1;
    }
    Reviewer *reviewers = (Reviewer *)malloc(MAX_REVIEWERS * sizeof(Reviewer));
    if (reviewers == NULL) {
        printf("Memory allocation failed\n");
        fclose(file);
        return 1;
    }
    int reviewerCount = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        char reviewerID[MAX_ID_LENGTH];
        char reviewerName[100];
        char reviewText[MAX_LINE_LENGTH];
        if (!extractField(line, "reviewerID", reviewerID, MAX_ID_LENGTH)) {
            continue;
        }
        
        extractField(line, "reviewerName", reviewerName, 100);
        
        if (!extractField(line, "reviewText", reviewText, MAX_LINE_LENGTH)) {
            continue;
        }

        int wordCount = countWords(reviewText);
        if (wordCount >= 50) {
            int index = findReviewer(reviewers, reviewerCount, reviewerID);
            
            if (index == -1) {
                // New reviewer
                if (reviewerCount < MAX_REVIEWERS) {
                    strcpy(reviewers[reviewerCount].reviewerID, reviewerID);
                    strcpy(reviewers[reviewerCount].reviewerName, reviewerName);
                    reviewers[reviewerCount].elaborateReviewCount = 1;
                    reviewerCount++;
                }
            } else {
                reviewers[index].elaborateReviewCount++;
            }
        }
    }
    
    printf("\n%-20s %-30s %s\n", "ReviewerID", "Reviewer Name", "Elaborate Review Count");
    printf("-------------------------------------------------------\n");
    
    int foundCount = 0;
    for (int i = 0; i < reviewerCount; i++) {
        if (reviewers[i].elaborateReviewCount >= 5) {
            printf("%-20s %-30s %d\n", reviewers[i].reviewerID, reviewers[i].reviewerName, 
                   reviewers[i].elaborateReviewCount);
            foundCount++;
        }
    }
    printf("\nPerformance Metrics, Ran in:%ld",startTime);
   
    printf("\nTotal: %d reviewers found\n", foundCount);

    free(reviewers);
    fclose(file);
    
    return 0;
}