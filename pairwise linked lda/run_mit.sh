#./lda est $1 $2 settings_fixed.txt ../raw_data/dataset/bow_imp_words_feature.txt ../raw_data/dataset/all.link seeded `echo ../raw_data/dataset/runs/k$2_a$1`
./lda est $1 $2 settings_fixed.txt ../CGL-dataset/mit/mit_bow_small.txt ../CGL-dataset/mit/mit_train.link seeded `echo ../run_CGL_small/mit/k$2_a$1`
