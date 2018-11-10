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

#include "lda-inference.h"

/*
 * variational inference
 *
 */

double lda_inference(document* doc, lda_model* model, 
        double* var_gamma, double** phi)
{
    double converged = 1;
    double phisum = 0, likelihood = 0;
    double likelihood_old = 0, oldphi[model->num_topics];
    int k, n, var_iter; 
    double digamma_gam[model->num_topics];

    // int d, d_;
    // double *lambda_sum, *lambda_sum_;
    // double lambda_converged = 1;
    // double lambda_old = 0;
    
    // compute posterior dirichlet

    for (k = 0; k < model->num_topics; k++)
    {
        var_gamma[k] = model->alpha + (doc->total/((double) model->num_topics));
        digamma_gam[k] = digamma(var_gamma[k]);
        for (n = 0; n < doc->length; n++)
            phi[n][k] = 1.0/model->num_topics;
    }
    var_iter = 0;

    while ((converged > VAR_CONVERGED) &&
           ((var_iter < VAR_MAX_ITER) || (VAR_MAX_ITER == -1)))
    {
    	var_iter++;
    	for (n = 0; n < doc->length; n++)
    	{
            phisum = 0;
            for (k = 0; k < model->num_topics; k++)
            {
                oldphi[k] = phi[n][k];
                phi[n][k] =
                    digamma_gam[k] +
                    model->log_prob_w[k][doc->words[n]];

                if (k > 0)
                    phisum = log_sum(phisum, phi[n][k]);
                else
                    phisum = phi[n][k]; // note, phi is in log space
            }

            for (k = 0; k < model->num_topics; k++)
            {
                phi[n][k] = exp(phi[n][k] - phisum);
                var_gamma[k] =
                    var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);
                // !!! a lot of extra digamma's here because of how we're computing it
                // !!! but its more automatically updated too.
                digamma_gam[k] = digamma(var_gamma[k]);
            }
        }

        likelihood = compute_likelihood(doc, model, phi, var_gamma);
        assert(!isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;

        // printf("[LDA INF] %8.5f %1.3e\n", likelihood, converged);
    }
    return(likelihood);
}


/*
 * compute likelihood bound
 *
 */

double
compute_likelihood(document* doc, lda_model* model, double** phi, double* var_gamma)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0, dig[model->num_topics];
    int k, n;

    for (k = 0; k < model->num_topics; k++)
    {
    	dig[k] = digamma(var_gamma[k]);
    	var_gamma_sum += var_gamma[k];
    }
    digsum = digamma(var_gamma_sum);

    likelihood =
	lgamma(model->alpha * model -> num_topics)
	- model -> num_topics * lgamma(model->alpha)
	- (lgamma(var_gamma_sum));

    for (k = 0; k < model->num_topics; k++)
    {
        likelihood +=
	    (model->alpha - 1)*(dig[k] - digsum) + lgamma(var_gamma[k])
	    - (var_gamma[k] - 1)*(dig[k] - digsum);

    	for (n = 0; n < doc->length; n++)
    	{
            if (phi[n][k] > 0)
            {
                likelihood += doc->counts[n]*
                    (phi[n][k]*((dig[k] - digsum) - log(phi[n][k])
                                + model->log_prob_w[k][doc->words[n]]));
            }
        }
    }
    return(likelihood);
}


/*
 * compute likelihood bound
 *
 */

double
compute_likelihood_pairwise_model(corpus* c, lda_model* model, 
                                 double*** phi, double** var_gamma, double*** lambda)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0, dig[model->num_topics];
    int k, n, d, d_, k_;
    document* doc;

    for (d = 0; d < c->num_docs; d++){
        doc = &(c->docs[d]);
        for (k = 0; k < model->num_topics; k++)
        {
            dig[k] = digamma(var_gamma[d][k]);
            var_gamma_sum += var_gamma[d][k];
        }
        digsum = digamma(var_gamma_sum);

        likelihood +=
        lgamma(model->alpha * model -> num_topics)
        - model -> num_topics * lgamma(model->alpha)
        - (lgamma(var_gamma_sum));

        if (isnan(likelihood))  printf("likelihood 1 %f\n", likelihood);

        for (k = 0; k < model->num_topics; k++)
        {
            likelihood +=
            (model->alpha - 1)*(dig[k] - digsum) + lgamma(var_gamma[d][k])
            - (var_gamma[d][k] - 1)*(dig[k] - digsum);

            for (n = 0; n < doc->length; n++)
            {
                if (phi[d][n][k] > 0)
                {
                    likelihood += doc->counts[n]*
                        (phi[d][n][k]*((dig[k] - digsum) - log(phi[d][n][k])
                                    + model->log_prob_w[k][doc->words[n]]));
                }
            }
        }
        if (isnan(likelihood))  printf("likelihood 2 %f\n", likelihood);
        for (d_ = 0; d_ < c->num_docs; d_++){
            for (k = 0; k < model->num_topics; k++)
            {
                for (k_ = 0; k_ < model->num_topics; k_++)
                {
                    likelihood += lambda[d][d_][k]*(dig[k] - digsum) + lambda[d_][d][k_]*(dig[k_] - digsum);
                    if (isnan(likelihood)){
                     printf("likelihood 3.1: %f\n", likelihood);
                     exit(0);
                    }
                    if (!(isnan(model->log_prob_n[k][k_]))){
                        if (c->citations[d][d_] > 0 )
                            likelihood += lambda[d][d_][k]*lambda[d_][d][k]*model->log_prob_n[k][k_];
                        else if (exp(model->log_prob_n[k][k_]) < 1){
                            likelihood += lambda[d][d_][k]*lambda[d_][d][k]*log(1-exp(model->log_prob_n[k][k_]));
                        }
                    }
                    if (isnan(likelihood)){
                     printf("likelihood 3.2: %f\n", likelihood);
                     printf("c->citations[%d][%d] %d model->log_prob_n[%d][%d] %f\n",d, d_,c->citations[d][d_], k, k_,model->log_prob_n[k][k_]);
                     exit(0);
                    }
                    if (lambda[d][d_][k] > 0)
                    {
                        likelihood -= lambda[d][d_][k]*log(lambda[d][d_][k]);
                    }
                    if (lambda[d_][d][k_] > 0)
                    {
                        likelihood -= lambda[d_][d][k_]*log(lambda[d_][d][k_]);
                    }
                }
            }
        }
    }
    return(likelihood);
}