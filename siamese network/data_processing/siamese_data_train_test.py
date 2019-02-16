import numpy as np
import cPickle
from sklearn.cross_validation import train_test_split
#import read_vec

class WORD(object):
    def __init__(self, d, n, dataset = 'University Course Dataset', pn='', datapath = '', test_perc=0.20):
        self.dim = d
	self.w, self.ind_wtrain, self.vocab = self.load_data(n, dataset, pn, datapath)
	#self.augment_data()
        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)
    
    def augment_data(self):
        neg = []
        indices = list(self.ind_wtrain)
        n = len(indices) - 1
	n_edges = len(self.y_train)
        np.random.seed(0)
        for _ in range(int(0.25*n_edges)):
            u = int(np.random.uniform() * n)
            neg.append((indices[u],indices[u]))
	for _ in range(int(0.75*n_edges)):
            rn = int(np.random.uniform() * (n_edges-1))
	    if self.y_train[rn] == 1:
	    	u, v = self.x_train[rn]
            	neg.append((v,u))
	self.x_train = self.x_train + neg
	self.y_train = self.y_train + [0]*len(neg)
    return 

    def get_gt(self, path, all_concepts):
        preqs = []
	labels = []
        with open(path, 'r') as f:
            for each in f:
                i,j,l = each.split(',')
                i = int(i)
                j = int(j)
		l = int(l)
                all_concepts.add(i)
                all_concepts.add(j)
                preqs.append((j, i))
		labels.append(l)
	data = {}
	for i in range(len(labels)):
	     data[preqs[i]] = labels[i]
	X = []
	Y = []
	for k in data.keys():
		X.append(k)
		Y.append(data[k])
	return X,Y, all_concepts
    #return preqs, labels, all_concepts

    def topics_per_word(self, indices, beta_file):
        word_topic = {}
        for i in indices:
            word_topic[i] = []
        for topic in file(beta_file, 'r'):
            # print 'topic %03d' % topic_no
            topic = map(float, topic.split())
	    n_w = len(topic)
            # print topic
            [word_topic[i].append(topic[i]) for i in indices if i<n_w]
        for i in word_topic.keys():
	    #vec = np.array(word_topic[i], dtype=np.float)
            vec = np.exp(np.array(word_topic[i], dtype=np.float))
            if len(vec) > 0: 
		mx = np.max(vec)
                if mx>0:
		    word_topic[i] = vec/mx
            #if mx != 0:
            #    word_topic[i] = vec/mx
            word_topic[i] = vec
        return word_topic, n_w

    # to remove the words in vocabulary which does not appear in text
    # and thus would not have any representation in beta matrix
    def remove_outofbeta_words(self, nw):
	print 'vocabulary size in bow ', nw
	c = 0
	l = len(self.y_train)
	i = 0
	while i<l-c:
	    u,v = self.x_train[i]
	    if (u >= nw or v >=nw):
		del self.x_train[i]
		del self.y_train[i]
		c = c+1
	    else:
		i = i+1
	print c, ' elements removed from train set'
	c = 0
        l = len(self.y_test)
        i = 0
        while i<l-c:
            u,v = self.x_test[i]
            if (u >= nw or v >=nw):
                del self.x_test[i]
                del self.y_test[i]
		c = c+1
            else:
                i = i+1
	print c, ' elements removed from test set'
	return

    def read_vocab(self, path):
	i = 0
	vocab = {}
	with open(path) as f:
	    for each in f:
	    	vocab[i] = each.strip()
		i = i +1
    	return vocab
    
    def load_data(self, n, dataset, pn, datapath):
        if dataset == 'University Course Dataset':
            with open('../datasets/University Course Dataset/concept_vocab.pkl', 'rb') as fid:
                vocab_ind_dict = cPickle.load(fid)
            all_concepts = set()
            self.x_train, self.y_train, all_concepts = self.get_gt('../datasets/University Course Dataset/train_test_5fold/train_'+str(n)+'_index.txt', all_concepts)
            self.x_test, self.y_test, word_inds = self.get_gt('../datasets/University Course Dataset/train_test_5fold/test_'+str(n)+'_index.txt', all_concepts)
            self.data_path = datapath +str(pn)+'/final.beta'
            words, n_w = self.topics_per_word(word_inds, self.data_path)
        elif dataset == 'NPTEL MOOC Dataset':
            with open('../datasets/NPTEL MOOC Dataset/concept_vocab.pkl', 'rb') as fid:
                vocab_ind_dict = cPickle.load(fid)
            all_concepts = set()
            self.x_train, self.y_train, all_concepts = self.get_gt('../datasets/NPTEL MOOC Dataset/train_test_5fold/train_'+str(n)+'_index.txt', all_concepts)
            self.x_test, self.y_test, word_inds = self.get_gt('../datasets/NPTEL MOOC Dataset/train_test_5fold/test_'+str(n)+'_index.txt', all_concepts)
            self.data_path = datapath +str(pn)+'/final.beta'
            words, n_w = self.topics_per_word(word_inds, self.data_path)
            self.remove_outofbeta_words(n_w)
        else:
            print "Unknown dataset. Mention path to beta file and vocab pickle. Exiting ..."
            exit(0)
        self.data_path = datapath +str(pn)+'/final.beta'
	return words, all_concepts, vocab_ind_dict

    
    def get_next_batch(self, batch, phase='train', one_shot=False):
        """
        Args:
            batch: an `integer` representing the size of the batch.
            phase: a `string` in `['train', 'test']`. Indicates which
                data to retrieve.
            one_shot: a `boolean`. Reads the whole set instead of random batches.

        Returns:
            x1_: a `numpy array` of shape `(batch, 28, 28)` containing
                images for the first network.
            x2_: a `numpy array` similar to x1_ containing images
                for the second network.
            y_: a `numpy array` of shape `(batch)` containing the labels.
        """
        
        x1_ = []
        x2_ = []
        y_ = []
        cs_ =[]
        ct_ = []
        if (one_shot and phase == "test"):
            for sn in range(self.n_test):
                u, v = self.x_test[sn]
                x1_.append(self.w[u])
                x2_.append(self.w[v])
                y_.append(self.y_test[sn])
		cs_.append(u)
		ct_.append(v)
        else:
            for _ in range(batch):
                if phase=="train":
                    sn = int(np.random.uniform() * (self.n_train-1)) 
                    u, v = self.x_train[sn]
                    x1_.append(self.w[u])
                    x2_.append(self.w[v])
                    y_.append(self.y_train[sn])
                    cs_.append(u)
                    ct_.append(v)
                else:
                    sn = int(np.random.uniform() * (self.n_test-1)) 
                    u, v = self.x_test[sn]
                    x1_.append(self.w[u])
                    x2_.append(self.w[v])
                    y_.append(self.y_test[sn])
                    cs_.append(u)
                    ct_.append(v)

        x1_ = np.asarray([x_.reshape((1, self.dim)) for x_ in x1_])
        x2_ = np.asarray([x_.reshape((1, self.dim)) for x_ in x2_])
        y_ = np.asarray(y_)
        cs_ = np.asarray([self.vocab[i] for i in cs_])
        ct_ = np.asarray([self.vocab[i] for i in ct_])
        return x1_, x2_, y_, cs_, ct_

