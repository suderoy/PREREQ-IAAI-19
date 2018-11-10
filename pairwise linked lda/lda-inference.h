#ifndef LDA_INFERENCE_H
#define LDA_INFERENCE_H

#include <math.h>
#include <float.h>
#include <assert.h>
#include "lda.h"
#include "utils.h"

float VAR_CONVERGED;
int VAR_MAX_ITER;

double lda_inference(document*, lda_model*, double*, double**);
double compute_likelihood(document*, lda_model*, double**, double*);
double compute_likelihood_pairwise_model(corpus* c, lda_model* model, 
                                 double*** phi, double** var_gamma, double*** lambda);
#endif
