#da est [initial alpha] [k] [settings] [data] [citation file] [random/seeded/manual=filename/*] [directory]
for i in {1..25..1}
  do
    preq=`echo ../citeseer/citations_pairwise_train.$i.txt`
    op=`echo ../citeseer/k100_a0.01_model.$i`
    echo lda est 0.01 100 settings.txt ../citeseer/doc_term_train.$i.txt $preq seeded $op
    ./lda est 0.01 100 settings.txt ../citeseer/doc_term_train.$i.txt $preq seeded $op
    echo lda inf settings.txt ../citeseer/k100_a0.01_model.$i  ../citeseer/citing_test.$i.doc_term.dat ../citeseer/citations_pairwise_test.$i.txt ../citeseer/k100_a0.01_test.$i
    ./lda inf settings.txt ../citeseer/k100_a0.01_model.$i  ../citeseer/citing_test.$i.doc_term.dat ../citeseer/citations_pairwise_test.$i.txt ../citeseer/k100_a0.01_test.$i
  done
