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

#include "lda-data.h"

corpus* read_data(char* data_filename)
{
    FILE *fileptr;
    int length, count, word, n, nd, nw;
    corpus* c;

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(corpus));
    c->docs = 0;
    c->num_terms = 0;
    c->num_docs = 0;
    fileptr = fopen(data_filename, "r");
    nd = 0; nw = 0;
    while ((fscanf(fileptr, "%10d", &length) != EOF))
    {
	c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
	c->docs[nd].length = length;
	c->docs[nd].total = 0;
	c->docs[nd].words = malloc(sizeof(int)*length);
	c->docs[nd].counts = malloc(sizeof(int)*length);
	for (n = 0; n < length; n++)
	{
	    fscanf(fileptr, "%10d:%10d", &word, &count);
	    word = word - OFFSET;
	    c->docs[nd].words[n] = word;
	    c->docs[nd].counts[n] = count;
	    c->docs[nd].total += count;
	    if (word >= nw) { nw = word + 1; }
	}
	nd++;
    }
    fclose(fileptr);
    c->num_docs = nd;
    c->num_terms = nw;
    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);

    return(c);
}

corpus* read_data_with_citation(char* data_filename, char* citations_filename)
{
    FILE *fileptr;
    int length, n, nd, nl, cited_doc;
    corpus* c = read_data(data_filename);

    printf("reading citations data from %s\n", citations_filename);
    c->citations = malloc(sizeof(int*)*c->num_docs);
    for (nd = 0; nd < c->num_docs; nd++){
        c->citations[nd] = malloc(sizeof(int)*c->num_docs);
        for (n = 0; n < c->num_docs; n++)
            c->citations[nd][n] = 0;
    }
    nl = 0;
    fileptr = fopen(citations_filename, "r");
    // nd = 0;
    // while((fscanf(fileptr, "%10d", &length) != EOF) && (nd < c->num_docs))
    // {
    //     c->num_citations += length;
    //     for (n = 0; n < length; n++)
    //     {
    //         fscanf(fileptr, "%10d", &cited_doc);
    //         c->citations[nd][cited_doc] = 1;
    //     }
    //     nd++;
    // }

    while((fscanf(fileptr, "%10d", &nd) != EOF))
    {
        nl++;
        fscanf(fileptr, "%10d ", &cited_doc);
        c->citations[nd-1][cited_doc-1] = 1;
    }
    fclose(fileptr);
    c->num_citations = nl;
    printf("number of links   : %d\n", nl);
    return (c);
}

int max_corpus_length(corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->num_docs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}
