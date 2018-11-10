import csv
import cPickle
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer() #this is a really bad stemmer

def lemmatize(text):
	text = unicode(text, errors='replace').strip().lower()
	t = ' '.join([st.stem(word) for word in text.split() if word not in (stopwords.words('english'))])
	return t

def strip_w(w):
	if w is None:
		return w
	else:
		return w.strip()

def get_ind(w):
	w1 = w.split('(')[0].lower().replace('_',' ').replace('-',' ').replace('/',' ')
	if vocab_dict_ori.get(w1) is not None:
		map_dict[w] = w1
		return vocab_dict_ori.get(w1)
	w1 = lemmatize(w1)
	map_dict[w] = vocab_dic.get(w1)
	return vocab_dict.get(w1)


vocab_dict = {}
vocab_dict_ori = {}
vocab_dic = {}
i = 0
with open('vocab.txt', 'r') as f:
	for w in f:
		vocab_dict_ori[w.strip().lower()] = i
		ww = lemmatize(w)
		# print ww
		vocab_dict[ww] = i
		vocab_dic[ww] = w
		i = i+1

concept_vocab = {}
of = open('cs_preqs_index.txt', 'w')
map_dict = {}
with open('cs_preqs.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
    for row in reader:
	    i1 = get_ind(row[0])
	    i2 = get_ind(row[1])
	    if (i1 is not None and i2 is not None):
	    	of.write(str(i1) + ','+ str(i2) + '\n')
	    	concept_vocab[i1] = row[0]
	    	concept_vocab[i2] = row[1]
	    # else:
	    # 	print "ERROR: Vocab not found!!\n", w1, str(vocab_dict.get(w1)), w2, str(vocab_dict.get(w2))
with open('concept_vocab.pkl', 'wb') as fid:
	cPickle.dump(concept_vocab, fid)
for k in map_dict.keys():
	print k.strip(), '  ', strip_w(map_dict[k])
# notfound = set()
# with open('cs_preqs.csv', 'rb') as f:
#     reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
#     for row in reader:
# 	    w1 = row[0].lower().strip()
# 	    w2 = row[1].lower().strip()
# 	    if vocab_dict.get(w1) is None:
# 	    	notfound.add(w1)
# 	    if vocab_dict.get(w2) is None:
# 	    	notfound.add(w2)
# print len(notfound)
# for x in notfound:
# 	print x
