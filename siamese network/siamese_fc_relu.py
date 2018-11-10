import matplotlib
matplotlib.use('Agg')
import sys
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from data_processing.siamese_data_train_test import WORD # load the data and process it
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics

# hyper parameters
max_iter = 3500 # maximum number of iterations for training
learning_rate = 0.0001 #0.001
batch_train = 128 # batch size for training
batch_test = 256 # batch size for testing
display = 100 # display the training loss and accuracy every `display` step
n_test = 500 # test the network every `n_test` step
summaries_dir = './siamese_summary'

# Architecture of the siamese network
n_inputs = int(sys.argv[3]) #100 # dimension of each of the input vectors
n_steps = 1 # sequence length
n_hidden = 512 # 64#128 #256 #128 # number of neurons of the bi-directional LSTM
n_classes = 2 # two possible classes, either `same` of `different`

x1 = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs]) # placeholder for the first network (concept 1)
x2 = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs]) # placeholder for the second network (concept 2)

# placeholder for the label. `1` for `same` and `0` for `different`.
y = tf.placeholder(tf.int64, shape=[None])

# placeholder for dropout (we could use different dropout for different part of the architecture)
keep_prob = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)
def reshape_input(x_):
    """
    Reshape the inputs to match the shape requirements of the function
    `tf.nn.bidirectional_rnn`
    
    Args:
        x_: a tensor of shape `(batch_size, n_steps, n_inputs)`
        
    Returns:
        A `list` of length `n_steps` with its elements being tensors of shape `(batch_size, n_inputs)`
    """
    x_ = tf.transpose(x_, [1, 0, 2]) # shape: (n_steps, batch_size, n_inputs)
    x_ = tf.split(x_, n_steps, 0) #tensor flow > 0.12
    #x_ = tf.split(0, n_steps, x_) # a list of `n_steps` tensors of shape (1, batch_size, n_steps)
    return [tf.squeeze(z, [0]) for z in x_] # remove size 1 dimension --> (batch_size, n_steps)
    
def add_fc(inputs, outdim, train_phase, scope_in):
    fc =  fully_connected(inputs, outdim, activation_fn=None, scope=scope_in + '/fc')
    fc_bnorm = tf.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                         training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.layers.dropout(fc_relu, rate= 0.1, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out

def embedding_model(feats, train_phase, scope_name,
                    fc_dim = n_inputs, embed_dim = n_hidden):
    """
        Build two-branch embedding networks.
        fc_dim: the output dimension of the first fc layer.
        embed_dim: the output dimension of the second fc layer, i.e.
                   embedding space dimension.
    """
    # each branch.
    fc1 = add_fc(feats, fc_dim, train_phase, scope_name)
    fc2 = fully_connected(fc1, embed_dim, activation_fn=None,
                               scope = scope_name + '_2')
    embed = tf.nn.l2_normalize(fc2, 1, epsilon=1e-10)
    return embed

x1_, x2_ = reshape_input(x1), reshape_input(x2)

with tf.variable_scope('siamese_network') as scope:
#     with tf.name_scope('Embed_1'):
        embed_1 = embedding_model(x1_, train_phase, 'Embed')
#     with tf.name_scope('Embed_2'):
        reuse=True
        scope.reuse_variables() # tied weights (reuse the weights)
        embed_2 = embedding_model(x2_, train_phase, 'Embed')

# Weights and biases for the layer that connects the outputs from the two networks
weights = tf.get_variable('weigths_out', shape=[n_hidden, n_classes],
                initializer=tf.random_normal_initializer(stddev=1.0/float(n_hidden)))
biases = tf.get_variable('biases_out', shape=[n_classes])

#last_state1 = tf.squeeze(embed_1)
#last_state2 = tf.squeeze(embed_2)
last_states_diff = tf.squeeze(tf.abs(embed_1 - embed_2), [0])
logits = tf.matmul(last_states_diff, weights) + biases

prediction = tf.nn.log_softmax(logits=logits)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf.reduce_mean(loss))
'''
global_step = tf.Variable(0, trainable=False)
init_learning_rate = 0.001
decay_step = 7 
decay_rate = 0.794
learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                       decay_step, decay_rate , staircase=True) #0.794
optim = tf.train.AdamOptimizer(init_learning_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = optim.minimize(loss, global_step=global_step)
'''
#correct_pred = tf.equal(tf.argmax(logits, 1), y) 
correct_pred = tf.equal(tf.argmax(prediction, 1), y) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#some tensor board stuff
with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(loss)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
test_writer = tf.summary.FileWriter(summaries_dir + '/test')

def calculate_precision_from_logits(y):
    n = len(y)
    count = 0
    for i in range(n):
        if y[i][2] == np.argmax(y[i][3]):
            count = count + 1
    return float(count)/float(n)

def calculate_recall_from_logits(y):
    n = len(y)
    count = 0
    total = 0
    for i in range(n):
        if y[i][2] == np.argmax(y[i][3]):
            count = count + y[i][2] 
        if y[i][2] == 1:
            total = total + 1
    return float(count)/float(total)

n_runs = 5 
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
dataset_name = sys.argv[1] #'University Course Dataset'
a= [''] #range(100, 900, 100)  #['']*50 #range(100, 1000, 100) #(100, 900, 100)
p50 = np.zeros(len(a))
p100 = np.zeros(len(a))
prec = 0.0 #np.zeros(len(a))
recall = 0.0 #np.zeros(len(a))
Fmeasure = 0.0
for n in range(n_runs):
    prec_50 = []
    prec_100 = []
    pre = []
    rec = []
    fm = []
    # load data
    # np.random.seed(0)
    # for n_preq in ['_preq100','_preq200','_preq300','_preq400','_preq500','_preq600','_preq700', '_preq800']:
    #     data = WORD(dataset='eaai', pn=n_preq) # load the data
    for n_preq in a:
        # data = WORD(dataset=dataset_name, pn=n_preq, datapath='../ACL2017-dataset/Sample_edges_previousedge_link/k100_a0.01_preq_')
        data = WORD(n_inputs, n, dataset=dataset_name, pn=n_preq, datapath=sys.argv[2])
        # examples_n = 10 # display some images
        # indexes = np.random.choice(range(len(data.y)), examples_n, replace=False)
        # for i in range(examples_n):
        #     u, v = data.x[indexes[i]]
        #     if data.y[indexes[i]] == 1:
        #         print data.vocab[u], '-->', data.vocab[v]
        #     else:
        #         print data.vocab[u], '-x->', data.vocab[v]
        print "total number of training examples = ", data.n_train
        print "total number of positive sample in train data = ", sum(data.y_train)
        print "total number of positive sample in test data = ", sum(data.y_test)
        print data.data_path

        #Train the netowrk
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init) # initialize all variables
            print('Network training begins.')
            for i in range(1, max_iter + 1):
                # We retrieve a batch of data from the training set
                batch_x1, batch_x2, batch_y, batch_cs, batch_ct = data.get_next_batch(batch_train, phase='train')
                # We feed the data to the network for training
                feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: .9, train_phase:True}
                _, loss_, accuracy_, summary = sess.run([optimizer, loss, accuracy, merged], feed_dict=feed_dict)
                
                if i % display == 0:
                    print('step %i, training loss: %.5f, training accuracy: %.3f' % (i, np.mean(loss_), accuracy_))
                train_writer.add_summary(summary, i)
                # Testing the network
                if i % n_test == 0:
                    # Retrieving data from the test set
                    batch_x1, batch_x2, batch_y, batch_cs, batch_ct = data.get_next_batch(batch_test, phase='test')
                    feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.0, train_phase:False}
                    accuracy_test, pred_test, logit_test, summary = sess.run([accuracy, prediction, logits, merged], feed_dict=feed_dict)
                    pred_prob = [p[1] for p in pred_test]
                    if any(np.isnan(pred_prob)):
                        print "nan pred prob for 1", logit_test
                    fpr, tpr, thresholds = roc_curve(batch_y, pred_prob)
                    roc_auc = auc(fpr, tpr)
                    print('testing step %i, accuracy %.3f, AUC %.3f' % (i, accuracy_test, roc_auc))
                    test_writer.add_summary(summary, i)
            print('********************************')
            print('Training finished.')   
            # testing the trained network on a large sample
            batch_x1, batch_x2, batch_y, batch_cs, batch_ct = data.get_next_batch(data.n_test, phase='test', one_shot=True)
            feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob:1.0, train_phase:False}
            accuracy_test, pred_test, logits_test, loss_test = sess.run([accuracy, prediction, logits, loss], feed_dict=feed_dict)
            print('********************************')
            print('Testing the network.')
            print('Network accuracy %.3f' % (accuracy_test))
            print('********************************')
            print 'Number of Test samples : ', len(batch_y)
            pred_prob = [p[1] for p in pred_test]

            fpr, tpr, thresholds = roc_curve(batch_y, pred_prob)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

                test_data = sorted(zip(batch_x1, batch_x2, batch_y, pred_prob, batch_cs, batch_ct), key=lambda x:x[3], reverse=True)
            print 'Len test data', len(test_data)
            print [(x[2], x[3], x[4], x[5]) for x in test_data[:5]]
            pat50 = sum([x[2] for x in test_data[:50]])/float(50)
            pat100 = sum([x[2] for x in test_data[:100]])/float(100)
            print "P@50 = ", pat50
            print "P@100 = ", pat100
            # for x in test_data[:100]:
            #     print x[2], x[3]
            test_data = sorted(zip(batch_x1, batch_x2, batch_y, logits_test, batch_cs, batch_ct), key=lambda x:x[3][1], reverse=True)
            print [(x[2], x[3], x[4], x[5]) for x in test_data[:5]]
            #pat50 = calculate_precision_from_logits(test_data[:50])
            #pat100 = calculate_precision_from_logits(test_data[:100])
            #print "P@50 = ", pat50
            #print "P@100 = ", pat100
            #pr = calculate_precision_from_logits(test_data)
            #re = calculate_recall_from_logits(test_data)
            #P = metrics.precision_score(batch_y, np.argmax(logits_test, 1))
            #R = metrics.recall_score(batch_y, np.argmax(logits_test, 1))
            #F1 = metrics.f1_score(batch_y, np.argmax(logits_test, 1))
            P = metrics.precision_score(batch_y, np.argmax(pred_test,1))
            R = metrics.recall_score(batch_y, np.argmax(pred_test,1))
            F1 = metrics.f1_score(batch_y, np.argmax(pred_test,1))
            pre.append(P)
            rec.append(R)
            fm.append(F1)

            tf.summary.scalar("P_50", pat50)
            tf.summary.scalar("P_100", pat100)
            prec_50.append(pat50)
            prec_100.append(pat100)

    p50 = p50 + np.array(prec_50)
    p100 = p100 + np.array(prec_100)
    max_ind = np.argmax(np.array(fm))
    prec = prec + np.array(pre[max_ind])
    recall = recall + np.array(rec[max_ind])
    Fmeasure = Fmeasure + np.array(fm[max_ind])

Fmeasure = Fmeasure/n_runs
p50 = p50/n_runs
p100 = p100/n_runs
prec = prec/n_runs
recall = recall/n_runs
file = open("fc_relu_auc_precision_"+dataset_name+".txt", 'a')
file.write(sys.argv[2] + '  max_iter =' + str(max_iter) + ' learning rate =' + str(learning_rate) + ' n_hidden = '+ str(n_hidden) +'\n') #'  decay_step ='+str(decay_step)+ '  decay_rate =' + str(decay_rate) +'\n')
file.write('Precision@50 : '+ str(p50)+'\n')
file.write('Precision@100 : '+ str(p100)+'\n')
file.write('Precision : '+ str(prec)+'\n')
file.write('Recall : '+ str(recall)+'\n')
file.write('F-score : '+ str(Fmeasure)+'\n')
file.write('AUC over iteration ; ' + str(aucs) + '\n')
fig = plt.figure()
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

file.write('mean false positive rate :'+ str(mean_fpr) + '\n')
file.write('mean true positive rate :'+ str(mean_tpr) + '\n')
file.write('mean AUC : '+ str(mean_auc) + '\n')
file.close()

std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on ' + dataset_name)
plt.legend(loc="lower right")
fig.savefig('fc_relu_ROC_on_' + dataset_name + '.png')
plt.show()
'''
fig = plt.figure()
plt.plot(a, p50, '-b', label='Pairwise LDA+Siamese')
plt.title("Precision@50")
fig.savefig("Precision_"+dataset_name+"small@50.png")
fig = plt.figure()
plt.plot(a, p100, '-b', label='Pairwise LDA+Siamese')
plt.title("Precision@100")
fig.savefig("Precision_"+dataset_name+"_small@100.png")
'''
