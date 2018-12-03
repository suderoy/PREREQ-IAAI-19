## This is the source code for the paper "Inferring Concept Prerequisite Relations from Online Educational Resources", Sudeshna Roy, Meghana Madhyastha, Sheril Lawrence, Vaibhav Rajan. 31st AAAI Conference on Innovative Applications of Artificial Intelligence (IAAI-19). If you use any part of this code cite this paper, https://arxiv.org/abs/1811.12640

### First run the pairwise Link LDA to learn the /beta matrix. This code is modfied from original LDA code.

$ cd "pairwise link lda"

$ make

$ ./lda est 0.01 100 settings.txt "../datasets/NPTEL MOOC Dataset/bow_feature.txt" "../datasets/NPTEL MOOC Dataset/prerequisites.link" random "../datasets/NPTEL MOOC Dataset/k100_a0.01"

$ cd ..

### Train the siamese network. It does a 10 fold cross-validation on the data

$ cd "siamese network"

$ python siamese_fc_relu.py  "NPTEL MOOC Dataset" "../datasets/NPTEL MOOC Dataset/k100_a0.01 100"


### You may change all the above occurence of "NPTEL MOOC Dataset" to "University Course Dataset" to run on University Course Dataset.


### to run on your own dataset

1. Have the concept vocabulary in a text file as "vocab.txt"

2. Have all the files in the raw data ready. The files are:

- *cs_courses.csv*: These are CS-related course information. Each line is formatted as "\<Course_id\>,\<Course_description\>". Note the course titles are located at the begining of the description.

- *cs_edges.csv*: There are course prerequisite information. Each line "\<course_1\>,\<course_2\>" represents \<course_2\> is a prerequisite for \<course_1\>.
  
- *cs_preqs.csv*: These are concept prerequisite pairs. Each line "\<Concept_A\>,\<Concept_B\>" represents the prerequisite relationship.


3. Mention the paths of "cs_preqs.csv" and "vocab.txt" in "data preprocessing/create_bow_features.py" and paths of "cs_course.csv" and "concept_vocab.pkl" in "data preprocessing/create_bow_features.py". Then run

$ python preq_preparation.py
$ python create_bow_features.py

4. Run the pairwise Link LDA with correct paths as explained earlier. 

5. Have the train-test splits ready and mention the data paths in the "siamese network/data_processing/siamese_data_train_test.py

6. Run the siamese network as explained above.
