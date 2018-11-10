// (C) Copyright 2004, David M. Blei (blei [at] cs [dot] cmu [dot] edu)

// This file is part of LDA-C.

// LDA-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// LDA-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "lda-estimate.h"

/*
 * perform inference on a document and update sufficient statistics
 *
 */

double doc_e_step(corpus* corpus, double** gamma, double*** phi, double*** lambda,
                             lda_model* model, lda_suffstats* ss, int e_iter)
{
    double likelihood = 0;
    int n, k, k_, d, d_, var_iter, cdd_, cd_d;
    double lambda_dd_, lambda_d_d;
    double lambda_converged = 1;
    double lambda_old = 0;
    double time_spent;

    // posterior inference
    for (d = 0; d < corpus->num_docs; d++)
    {
        likelihood += lda_inference(&(corpus->docs[d]), model, gamma[d], phi[d]);

    }
    printf("Step 1 likelihood %f\n", likelihood);
    clock_t timer, now, begin;
    begin = clock();
    for (d = 0; d < corpus->num_docs; d++)
    {
        for (d_ = d+1; d_ < corpus->num_docs; d_++)
        {
	    timer = clock();
            var_iter = 0;
            lambda_converged = 1;
            cdd_ = corpus->citations[d][d_];
            cd_d = corpus->citations[d_][d];
            
            while ((lambda_converged > VAR_CONVERGED) &&
                ((var_iter < VAR_MAX_ITER) || (VAR_MAX_ITER == -1))){
                var_iter++;

                lambda_dd_ = 0;
                lambda_d_d = 0;
                for (k = 0; k < model->num_topics; k++){
                    lambda[d][d_][k] = digamma(gamma[d][k]);
                    for (k_ = 0; k_ < model->num_topics; k_++){
                        if (!(isnan(model->log_prob_n[k][k_]))){
                            if (cdd_ > 0){
                                lambda[d][d_][k] += lambda[d_][d][k_] * model->log_prob_n[k][k_];
                            }
                            else if (exp(model->log_prob_n[k][k_]) < 1){
                                lambda[d][d_][k] += lambda[d_][d][k_] * log(1-exp(model->log_prob_n[k][k_]));
                            }
                        }
                    }
                    lambda[d][d_][k] = exp(lambda[d][d_][k]);
                    //lambda_dd_ = log_sum(lambda_dd_ ,lambda[d][d_][k]); // lambda is in log space
                    lambda_dd_ += lambda[d][d_][k];
                }
                for (k = 0; k < model->num_topics; k++){
                    if (lambda_dd_ > 0)
                    lambda[d][d_][k] = lambda[d][d_][k]/lambda_dd_;
                    //printf("lambda[%d][%d][%d] = %f, sum %f\n", d, d_, k, lambda[d][d_][k], lambda_dd_);
                }

                for (k = 0; k < model->num_topics; k++){
                    lambda[d_][d][k] = digamma(gamma[d_][k]);
                    //printf("lambda[%d][%d][%d] = %f\n", d_,d,k, lambda[d_][d][k]);
                    for (k_ = 0; k_ < model->num_topics; k_++){
                        if (!(isnan(model->log_prob_n[k][k_]))){
                            if (cd_d > 0){
                                lambda[d_][d][k] += lambda[d][d_][k_] * model->log_prob_n[k][k_];
                            }
                            else if (exp(model->log_prob_n[k][k_]) < 1){
                                lambda[d_][d][k] += lambda[d][d_][k_] * log(1-exp(model->log_prob_n[k][k_]));
                            }
                        }
                    }
                    // lambda_d_d = log_sum(lambda_d_d, lambda[d_][d][k]);
                    lambda[d_][d][k] = exp(lambda[d_][d][k]);
                    lambda_d_d += lambda[d_][d][k];
                }
                for (k = 0; k < model->num_topics; k++){
                    if(lambda_d_d > 0)
                    lambda[d_][d][k] = lambda[d_][d][k] /lambda_d_d;
                    //printf("lambda[%d][%d][%d] = %f, sum %f\n", d_, d, k, lambda[d_][d][k], lambda_d_d);
                }

                // lambda_converged = ( lambda_dd_- lambda_d_d)/(lambda_dd_ + lambda_d_d);

                lambda_converged = ( lambda_old - lambda_dd_ - lambda_d_d)/lambda_old;
                lambda_old =  lambda_dd_ + lambda_d_d;
                if (e_iter == 1)    break;
	    }
	    now = clock();
            printf("time elasped %f for %d, %d pair\n", (double)(now-timer), d, d_);
	    
            for (k = 0; k < model->num_topics; k++){
                //printf("lambda converged : lambda[%d][%d][%d] = %f\n", d, d_, k, lambda[d][d_][k]);
                if (isnan(lambda[d][d_][k]) || lambda[d][d_][k] < 0)
                printf("lambda converged BUT!!!: lambda[%d][%d][%d] = %f\n", d, d_, k, lambda[d][d_][k]);
                // exit(0);
            }

        }
    }
    now = clock();
            printf("time elasped %f for %d, %d pair\n", (double)(now-begin), d, d_);
    begin = clock();
    // update gamma
    for (d = 0; d < corpus->num_docs; d++)
    {
        for (k = 0; k < model->num_topics; k++)
        {
            for (d_ = 0; d_ < corpus->num_docs; d_++)
            {
                gamma[d][k] += lambda[d][d_][k];
            }
            if (isnan(gamma[d][k]) || gamma[d][k] <=0){
                printf("gamma[%d][%d] = %f\n", d, k, gamma[d][k]);
                //exit(0);
            }
        }
    }
    now = clock();
    printf("gamma update time elasped %f\n", (double)(now-begin));

    begin = clock();
    likelihood = compute_likelihood_pairwise_model(corpus, model, phi, gamma, lambda);
    printf("likelihood of pairwise model %f\n", likelihood);
    now = clock();
    printf("pairwise likelihood computation time elasped %f\n", (double)(now-begin));

    begin = clock();

    // update sufficient statistics
    ss->num_docs = corpus->num_docs;
    for (d = 0; d < corpus->num_docs; d++)
    {
        double gamma_sum = 0;
        for (k = 0; k < model->num_topics; k++)
        {
            gamma_sum += gamma[d][k];
            ss->alpha_suffstats += digamma(gamma[d][k]);
        }
        ss->alpha_suffstats -= model->num_topics * digamma(gamma_sum);
        document* doc = &(corpus->docs[d]);
        for (n = 0; n < doc->length; n++)
        {
            for (k = 0; k < model->num_topics; k++)
            {
                ss->class_word[k][doc->words[n]] += doc->counts[n]*phi[d][n][k];
                ss->class_total[k] += doc->counts[n]*phi[d][n][k];
            }
        }
    }

    for (k = 0; k < model->num_topics; k++)
    {
        for (k_ = 0; k_ < model->num_topics; k_++)
        {
            ss->eta_suffstats_total[k][k_] = 0;
            ss->eta_suffstats[k][k_] = 0;
            for (d = 0; d < corpus->num_docs; d++)
            {
                for (d_ = 0; d_ < corpus-> num_docs; d_++){
                    ss->eta_suffstats_total[k][k_] += lambda[d][d_][k] * lambda[d_][d][k_];
                    ss->eta_suffstats[k][k_] += lambda[d][d_][k] * lambda[d_][d][k_] * corpus->citations[d][d_]; 
                }
            }
        }
    }
    now = clock();
    printf("sufficient stat update time elasped %f \n", (double)(now-begin));
    return(likelihood);
}


/*
 * writes the word assignments line for a document to a file
 *
 */

void write_word_assignment(FILE* f, document* doc, double** phi, lda_model* model)
{
    int n;

    fprintf(f, "%03d", doc->length);
    for (n = 0; n < doc->length; n++)
    {
        fprintf(f, " %04d:%02d",
                doc->words[n], argmax(phi[n], model->num_topics));
    }
    fprintf(f, "\n");
    fflush(f);
}

void write_doc_topic_dist(FILE* f, document* doc, lda_model* model)
{
    int n, k;
    double* topic_dist = (double*)malloc(sizeof(double)*model->num_topics);
    double sum = 0;
    for (k = 0; k < model->num_topics; k++){
        for (n = 0; n < doc->length; n++)
        {
            topic_dist[k] += model->log_prob_w[k][doc->words[n]];
        }
        sum += topic_dist[k];
    }
    for (k = 0; k < model->num_topics; k++){
        topic_dist[k] /= sum;
        fprintf(f, "%f ", topic_dist[k]);
    }
    fprintf(f, "\n");
    fflush(f);
}


/*
 * saves the gamma parameters of the current dataset
 *
 */

void save_gamma(char* filename, double** gamma, int num_docs, int num_topics)
{
    FILE* fileptr;
    int d, k;
    fileptr = fopen(filename, "w");

    for (d = 0; d < num_docs; d++)
    {
	fprintf(fileptr, "%5.10f", gamma[d][0]);
	for (k = 1; k < num_topics; k++)
	{
	    fprintf(fileptr, " %5.10f", gamma[d][k]);
	}
	fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}


/*
 * run_em
 *
 */

void run_em(char* start, char* directory, corpus* corpus)
{

    int d,d_,n,k;
    lda_model *model = NULL;
    double **var_gamma, ***phi, ***lambda;

    // allocate variational parameters

    var_gamma = malloc(sizeof(double*)*(corpus->num_docs));
    for (d = 0; d < corpus->num_docs; d++)
	   var_gamma[d] = malloc(sizeof(double) * NTOPICS);

    int max_length = max_corpus_length(corpus);
    phi = malloc(sizeof(double**)*(corpus->num_docs));
    for (d = 0; d < corpus->num_docs; d++){
        phi[d] = malloc(sizeof(double*)*max_length);
        for (n = 0; n < max_length; n++)
	       phi[d][n] = malloc(sizeof(double) * NTOPICS);
    }
    
    lambda = malloc(sizeof(double**)*(corpus->num_docs));
    for (d = 0; d < corpus->num_docs; d++){
        lambda[d] = malloc(sizeof(double*)*(corpus->num_docs));
        for (d_ = 0; d_ < corpus->num_docs; d_++){
            lambda[d][d_] = malloc(sizeof(double) * NTOPICS);
        }
    }

    for (d = 0; d < corpus->num_docs; d++)
    {
        for (d_ = 0; d_ < corpus->num_docs; d_++)
        {
            for (k = 0; k < NTOPICS; k++){
                lambda[d][d_][k] = (double)rand()/(double)RAND_MAX;
                //printf("lambda[%d][%d][%d] = %f\n", d,d_,k, lambda[d][d_][k]);
            }
        }
    }

    // initialize model

    char filename[100];

    lda_suffstats* ss = NULL;
    if (strcmp(start, "seeded")==0)
    {
        model = new_lda_model(corpus->num_terms, NTOPICS);
        ss = new_lda_suffstats(model);
        corpus_initialize_ss(ss, model, corpus);
        lda_mle(model, ss, 0);
        model->alpha = INITIAL_ALPHA;
    }
    else if (strcmp(start, "random")==0)
    {
        model = new_lda_model(corpus->num_terms, NTOPICS);
        ss = new_lda_suffstats(model);
        random_initialize_ss(ss, model);
        lda_mle(model, ss, 0);
        model->alpha = INITIAL_ALPHA;
    }
    else if (strncmp(start, "manual=",7)==0)
    {
        model = new_lda_model(corpus->num_terms, NTOPICS);
        ss = new_lda_suffstats(model);
        manual_initialize_ss(start + 7, ss, model, corpus);
        lda_mle(model, ss, 0);
        model->alpha = INITIAL_ALPHA;
    }
    else
    {
        model = load_lda_model(start);
        ss = new_lda_suffstats(model);
    }

    sprintf(filename,"%s/000",directory);
    save_lda_model(model, filename);

    // run expectation maximization

    int i = 0;
    double likelihood, likelihood_old = 0, converged = 1;
    sprintf(filename, "%s/likelihood.dat", directory);
    FILE* likelihood_file = fopen(filename, "w");

    while (((converged < 0) || (converged > EM_CONVERGED) || (i <= 2)) && (i <= EM_MAX_ITER))
    {
        i++; printf("**** em iteration %d ****\n", i);
        likelihood = 0;
        zero_initialize_ss(ss, model);

        // e-step

        likelihood = doc_e_step(corpus,
                                 var_gamma,
                                 phi,
                                 lambda,
                                 model,
                                 ss, i);
        

        // m-step

        lda_mle(model, ss, ESTIMATE_ALPHA);

        // check for convergence

        converged = (likelihood_old - likelihood) / (likelihood_old);
        if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2;
        likelihood_old = likelihood;

        // output model and likelihood

        fprintf(likelihood_file, "%10.10f\t%5.5e\n", likelihood, converged);
        fflush(likelihood_file);
        if ((i % LAG) == 0)
        {
            sprintf(filename,"%s/%03d",directory, i);
            save_lda_model(model, filename);
            sprintf(filename,"%s/%03d.gamma",directory, i);
            save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);
        }
    }

    // output the final model

    sprintf(filename,"%s/final",directory);
    save_lda_model(model, filename);
    sprintf(filename,"%s/final.gamma",directory);
    save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);

    // output the word assignments (for visualization)

    sprintf(filename, "%s/word-assignments.dat", directory);
    FILE* w_asgn_file = fopen(filename, "w");
    for (d = 0; d < corpus->num_docs; d++)
    {
        if ((d % 100) == 0) printf("final e step document %d\n",d);
        likelihood += lda_inference(&(corpus->docs[d]), model, var_gamma[d], phi[d]);
        write_word_assignment(w_asgn_file, &(corpus->docs[d]), phi[d], model);
    }
    fclose(w_asgn_file);
    sprintf(filename, "%s/doc-topic-dist.dat", directory);
    FILE* doc_topic_file = fopen(filename, "w");
    for (d = 0; d < corpus->num_docs; d++)
    {
        write_doc_topic_dist(doc_topic_file, &(corpus->docs[d]), model);
    }
    fclose(doc_topic_file);
    fclose(likelihood_file);
}


/*
 * read settings.
 *
 */

void read_settings(char* filename)
{
    FILE* fileptr;
    char alpha_action[100];
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "var max iter %d\n", &VAR_MAX_ITER);
    fscanf(fileptr, "var convergence %f\n", &VAR_CONVERGED);
    fscanf(fileptr, "em max iter %d\n", &EM_MAX_ITER);
    fscanf(fileptr, "em convergence %f\n", &EM_CONVERGED);
    fscanf(fileptr, "alpha %s", alpha_action);
    if (strcmp(alpha_action, "fixed")==0)
    {
	ESTIMATE_ALPHA = 0;
    }
    else
    {
	ESTIMATE_ALPHA = 1;
    }
    fclose(fileptr);
}


/*
 * inference only
 *
 */

void infer(char* model_root, char* save, corpus* corpus)
{
    FILE* fileptr;
    char filename[100];
    int i, d, n;
    lda_model *model;
    double **var_gamma, likelihood, **phi;
    document* doc;

    model = load_lda_model(model_root);
    var_gamma = malloc(sizeof(double*)*(corpus->num_docs));
    for (i = 0; i < corpus->num_docs; i++)
	var_gamma[i] = malloc(sizeof(double)*model->num_topics);
    sprintf(filename, "%s-lda-lhood.dat", save);
    fileptr = fopen(filename, "w");
    for (d = 0; d < corpus->num_docs; d++)
    {
	if (((d % 100) == 0) && (d>0)) printf("document %d\n",d);

	doc = &(corpus->docs[d]);
	phi = (double**) malloc(sizeof(double*) * doc->length);
	for (n = 0; n < doc->length; n++)
	    phi[n] = (double*) malloc(sizeof(double) * model->num_topics);
	likelihood = lda_inference(doc, model, var_gamma[d], phi);

	fprintf(fileptr, "%5.5f\n", likelihood);
    }
    fclose(fileptr);
    sprintf(filename, "%s-gamma.dat", save);
    save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);
}

void predict(char* model_root, char* save, corpus* corpus)
{
    FILE* fileptr;
    FILE* p_fileptr;
    char filename[100];
    char predict_filename[100];
    int i, d, n, d_, k, k_;
    lda_model *model;
    double **var_gamma, *var_gamma_sum, var_gamma_sum_d, likelihood, **phi, p_d;
    document* doc;

    model = load_lda_model(model_root);
    var_gamma = malloc(sizeof(double*)*(corpus->num_docs));
    for (i = 0; i < corpus->num_docs; i++)
       var_gamma[i] = malloc(sizeof(double)*model->num_topics);
    var_gamma_sum = malloc(sizeof(double*)*(corpus->num_docs));
    sprintf(filename, "%s-lda-lhood.dat", save);
    fileptr = fopen(filename, "w");
    for (d = 0; d < corpus->num_docs; d++)
    {
        if (((d % 100) == 0) && (d>0)) printf("document %d\n",d);

        doc = &(corpus->docs[d]);
        phi = (double**) malloc(sizeof(double*) * doc->length);
        for (n = 0; n < doc->length; n++)
           phi[n] = (double*) malloc(sizeof(double) * model->num_topics);
        likelihood = lda_inference(doc, model, var_gamma[d], phi);
        fprintf(fileptr, "%5.5f\n", likelihood);
        var_gamma_sum_d = 0;
        for(k = 0; k < model->num_topics;k++)
            var_gamma_sum_d += var_gamma[d][k];
        var_gamma_sum[d] = var_gamma_sum_d;

        for(n = 0; n < doc->length;n++)
            free(phi[n]);
        free(phi);
    }
    fclose(fileptr);
    sprintf(predict_filename, "%s-pred_scores.dat", save);
    p_fileptr = fopen(predict_filename, "w");
    sprintf(filename, "%s-argmax_pred.dat", save);
    fileptr = fopen(filename, "w");
    for (d = 0; d < corpus->num_docs; d++)
    {
        double max = 0;
        int dlink=-1;
        if (((d % 100) == 0) && (d>0)) printf("Computing links with document %d\n",d);
        for (d_ = 0; d_ < corpus->num_docs; d_++)
        {
            p_d = isnan(model->log_prob_n[0][0])?0:exp(model->log_prob_n[0][0])
             + var_gamma[d][0]/var_gamma_sum[d] + var_gamma[d_][0]/var_gamma_sum[d_];
            for (k = 0; k < model->num_topics; k++)
            {
                for (k_ = 0; k_ < model->num_topics; k_++)
                {
                    p_d = p_d + isnan(model->log_prob_n[0][0])?0:exp(model->log_prob_n[0][0]) + 
                        var_gamma[d][k]/var_gamma_sum[d] + var_gamma[d_][k_]/var_gamma_sum[d_];
                }
            }
            fprintf(p_fileptr, "%3d, %3d : %5.5f\n",d,d_,p_d);
            if (p_d > max){
                max = p_d;
                dlink = d_;
            }
        }
        fprintf(fileptr, "%3d, %3d : %5.5f\n",d,dlink,max);
    }
    fclose(fileptr);
    fclose(p_fileptr);
    sprintf(filename, "%s-gamma.dat", save);
    save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);
    for(d = 0; d < corpus->num_docs; d++)
        free(var_gamma[d]);
    free(var_gamma);
    free(var_gamma_sum);
}
/*
 * update sufficient statistics
 *
 */



/*
 * main
 *
 */

int main(int argc, char* argv[])
{
    // (est / inf) alpha k settings data (random / seed/ model) (directory / out)

    corpus* corpus;

    long t1;
    (void) time(&t1);
    seedMT(t1);
    // seedMT(4357U);

    if (argc > 1)
    {
        if (strcmp(argv[1], "est")==0)
        {
            INITIAL_ALPHA = atof(argv[2]);
            NTOPICS = atoi(argv[3]);
            read_settings(argv[4]);
            corpus = read_data_with_citation(argv[5], argv[6]);
            make_directory(argv[8]);
            run_em(argv[7], argv[8], corpus);
        }
        if (strcmp(argv[1], "inf")==0)
        {
            read_settings(argv[2]);
            corpus = read_data(argv[4]);
            infer(argv[3], argv[5], corpus);
        }
        if (strcmp(argv[1], "pred")==0)
        {
            read_settings(argv[2]);
            corpus = read_data(argv[4]);
            predict(argv[3], argv[5], corpus);
        }
    }
    else
    {
        printf("usage : lda est [initial alpha] [k] [settings] [data] [citation file] [random/seeded/manual=filename/*] [directory]\n");
        printf("        lda inf [settings] [model] [data] [name]\n");
	printf("        lda pred [settings] [model] [data] [name]\n");
    }
    return(0);
}
