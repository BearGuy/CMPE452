#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define LEARNING_RATE    0.5
#define MAX_ITERATION    5

float randomFloat()
{
    return (float)rand() / (float)RAND_MAX;
}

int calculateOutput(float weights[], float x[])
{
    float sum = x[1] * weights[1] + x[2] * weights[2] + x[3] * weights[3] + x[4] * weights[4];
    return (sum >= weights[0]) ? 1 : 0;
}
/*

input data X {x1, x2, x3, x4} ouput D {d1} => directly maps to different classes -1, 0, 1

*/
int main() {
    srand(time(NULL));

    // train
    float localError, globalError;
    float x[120][5], y[30][5];
    float weights[4][5];
    int outputs[120][3], patternCount, i, p, h, a, b, d, iteration, output;
    int theo_outputs[123][3], calc_outputs[123][3];
    char train_outputs[120][16], test_outputs[120][16];

    FILE *fp;
    if ((fp = fopen("./data/train.txt", "r")) == NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }

    i = 0;
    while (fscanf(fp, "%f%*c %f%*c %f%*c %f%*c %s", &x[i][1], &x[i][2], &x[i][3], &x[i][4], &train_outputs[i]) != EOF) {
        if (strcmp(train_outputs[i],"Iris-setosa") == 0) {
            outputs[i][0] = 1;
            outputs[i][1] = 0;
            outputs[i][2] = 0;
        }
        if (strcmp(train_outputs[i],"Iris-versicolor") == 0) {
            outputs[i][0] = 0;
            outputs[i][1] = 1;
            outputs[i][2] = 0;
        }
        if (strcmp(train_outputs[i],"Iris-virginica") == 0) {
            outputs[i][0] = 0;
            outputs[i][1] = 0;
            outputs[i][2] = 1;
        }
        x[i][0] = 1;
        i++;
    }
    patternCount = i;

    for (int a = 0; a < 3; a++) {
        for(int b = 0; b < 5; b++) {
            weights[a][b] = randomFloat();
        }
    }

    iteration = 0;
    do {
        iteration++;
        globalError = 0;
        for (p = 0; p < patternCount; p++) {
            for(h = 0; h < 3; h++) {
                output = calculateOutput(weights[h], x[p]);
                localError = outputs[p][h] - output;
                globalError += localError;
                
                // if (h == 1) {
                //     printf("Iteration: %d\n", iteration);
                //     printf("Calculated Output: %d\n", output);
                //     printf("Theoretical Output: %d\n", outputs[p][h]);
                //     printf("Global Error: %f\n", globalError);
                // }
                
                if (output > outputs[p][h]) {
                    for(d = 0; d < 5; d++){
                        weights[h][d] = weights[h][d] - LEARNING_RATE*x[p][d];
                    }
                }
                else if(output < outputs[p][h]) {
                    for(d = 0; d < 5; d++){
                        weights[h][d] = weights[h][d] + LEARNING_RATE*x[p][d];
                    }
                } 
            }
        }
    } while (globalError != 0 && iteration <= MAX_ITERATION);
    printf("Iteration: %d\n", iteration);

    // test
    FILE *op;
    if ((op = fopen("./data/test.txt", "r")) == NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }

    i = 0;
    while (fscanf(op, "%f%*c %f%*c %f%*c %f%*c %s", &y[i][1], &y[i][2], &y[i][3], &y[i][4], &test_outputs[i]) != EOF) {
        if (strcmp(test_outputs[i], "Iris-setosa") == 0) {
            theo_outputs[i][0] = 1;
            theo_outputs[i][1] = 0;
            theo_outputs[i][2] = 0;
        }
        else if (strcmp(test_outputs[i],"Iris-versicolor") == 0){
            theo_outputs[i][0] = 0;
            theo_outputs[i][1] = 1;
            theo_outputs[i][2] = 0;
        }
        else if (strcmp(test_outputs[i], "Iris-virginica") == 0){
            theo_outputs[i][0] = 0;
            theo_outputs[i][1] = 0;
            theo_outputs[i][2] = 1;
        }
        y[i][0] = 1;
        i++;
    }
    patternCount = i;
    float success = 0;
    
    float scoring[2][3];

    for (p = 0; p < patternCount; p++) {
        int predictSuccess = 0;
        for(h = 0; h < 3; h++) {
            calc_outputs[p][h] = calculateOutput(weights[h], y[p]);

            if (calc_outputs[p][h] == theo_outputs[p][h]) {
                predictSuccess++;
            }

            //printf("calc_output: %d, theo_output: %d\n", calc_outputs[p][h], theo_outputs[p][h]);
            
            // Precision = True Class A / (TA + FA)
            // Recall = TA / (TA + FB + FC)

            if (predictSuccess == 3) {
                success += 1;
                predictSuccess = 0;
            }
            else if(h == 2){
                predictSuccess = 0;
            }
        }

        if (calc_outputs[p][0] == 1 && theo_outputs[p][0] == 1) {
            scoring[0][0] += 1;
        }
        else if (calc_outputs[p][0] == 1 && theo_outputs[p][0] == 0) {
            scoring[1][0] += 1;
        }
        else if (calc_outputs[p][1] == 1 && theo_outputs[p][1] == 1) {
            scoring[0][1] += 1;
        }
        else if (calc_outputs[p][1] == 1 && theo_outputs[p][0] == 0) {
            scoring[1][1] += 1;
        }
        else if (calc_outputs[p][2] == 1 && theo_outputs[p][2] == 1) {
            scoring[0][2] += 1;
        }
        else if (calc_outputs[p][2] == 1 && theo_outputs[p][2] == 0) {
            scoring[1][2] += 1;
        }
    }

    float score = success / 30 ;
    
    float precisionA = scoring[0][0] / (scoring[0][0] + scoring[1][0]);
    float recallA = scoring[0][0] / (scoring[0][0] + scoring[1][1] + scoring[1][2]);

    float precisionB = scoring[0][1] / (scoring[0][1] + scoring[1][1]) || 0;
    float recallB = scoring[0][1] / (scoring[0][1] + scoring[1][0] + scoring[1][2]);

    float precisionC = scoring[0][2] / (scoring[0][2] + scoring[1][2]);
    float recallC = scoring[0][2] / (scoring[0][2] + scoring[1][1] + scoring[1][0]);

    printf("Precision A: %f\n", precisionA);
    printf("Recall A: %f\n", recallA);

    printf("Precision B: %f\n", precisionB);
    printf("Recall B: %f\n", recallB);

    printf("Precision C: %f\n", precisionC);
    printf("Recall C: %f\n", recallC);

    printf("Final Score: %f\n", score);
    
    return 0;
}