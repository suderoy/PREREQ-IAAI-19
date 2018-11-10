from sklearn.feature_extraction.text import CountVectorizer
import io
import subprocess
import os
import csv
import pickle
def create_bow(course_text, vocab=[]):
    print "Creating the bag of words...\n"
    vectorizer = CountVectorizer(analyzer = "word",   \
                             ngram_range=(1, 3),
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = "english",
                             vocabulary = vocab)#,   \
                             #max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(course_text)
    train_data_features = train_data_features.toarray()
    write_unicode_file(reduce(lambda x ,y: x+y+unicode('\n'), vectorizer.get_feature_names(), unicode('')), 'vocab.txt')
    return train_data_features


def write_bow(data):
    file = open('bow_feature.txt', 'w')
    for each in data:
        w = each.nonzero()[0]
        bow_str = reduce(lambda x,y: x+' '+str(y)+':'+str(each[y]), w, str(len(w)))
        file.write(bow_str + '\n')

def write_links(file, path, id_dict):
    of = open(path,'w')
    with open(file, 'rb') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in reader:
            a = row[0]
            b = row[1]
            print a,b
            print id_dict.get(a), id_dict.get(b) 
            if (id_dict.get(a) is not None and id_dict.get(b) is not None): 
                of.write(str(id_dict.get(a)) + ' ' + str(id_dict.get(b)) + '\n')
    of.close()
    
def write_unicode_file(d, p):
    f = io.open(p, 'w', encoding='utf-8')
    f.write(unicode(d))
    f.close()

def write_file(d, p):
    f = open(p, 'w' )
    f.write(str(d))
    f.close()

def process_vocab(l):
    v = []
    for e in l:
        x = e.split('(')[0]
	x = x.replace('_',' ').lower()
	v.append(x)
    return v

all_text = []
id_dict = {}
index = 1
with open('cs_courses.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
    for row in reader:    
        id_dict[row[0]] = index
        index = index + 1
        text = row[1]
        all_text.append(text)
vocab = process_vocab(pickle.load(open('concept_vocab.pkl')).values())
train_data_features = create_bow(all_text, vocab)
write_bow(train_data_features)
write_file(id_dict, 'id_map.txt')
write_links('cs_edges.csv', 'prerequisites.link', id_dict)

# id_dict = eval(open('id_map.txt').read())
# for i in range(100, 900, 100):
#     write_links('data/'+str(i)+'.csv', 'data/prerequisites_'+str(i)+'.link', id_dict)
