from __future__ import division, unicode_literals
import numpy as np
import sys,os
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

try:
    unicode("is this python3")
except NameError:
    unicode = str
    xrange = range
    
from IPython import embed

vocabfile = "model_vocab.txt"
modelfile = "model_count_table.txt"

if not os.path.isfile(vocabfile):
    dn = os.path.dirname(__file__)
    vocabfile = os.path.join(dn,vocabfile)
    modelfile = os.path.join(dn,modelfile)

K=0; wordprobs=None; w2num=None

def load_model():
    """Idempotent"""
    global vocab,w2num,N_wk,N_k,wordprobs,N_w,K, modelfile,vocabfile
    if wordprobs is not None:
        # assume already loaded
        return

    N_wk = np.loadtxt(modelfile)
    N_w = N_wk.sum(1)
    N_k = N_wk.sum(0)

    K = len(N_k)
    wordprobs = (N_wk + 1) / N_k


    # vocab = [L.split("\t")[-1].strip().decode("utf-8") for L in open(vocabfile)]
    vocab = [L.split("\t")[-1].strip() for L in open(vocabfile)]
    w2num = {w:i for i,w in enumerate(vocab)}
    assert len(vocab) == N_wk.shape[0]

def infer_cvb0(invocab_tokens, alpha, numpasses):
    global K,wordprobs,w2num
    
    doclen = len(invocab_tokens)
    
    # initialize with likelihoods
    Qs = np.zeros((doclen, K))
    for i in xrange(doclen):
        w = invocab_tokens[i]
        Qs[i,:] = wordprobs[w2num[w],:]
        Qs[i,:] /= Qs[i,:].sum()
        # if Qs[i,0]/Qs[i,:].sum() > 0.50:
        #     Qs[i,0] = Qs[i,0]*10
    lik = Qs.copy()  # pertoken normalized but proportionally the same for inference

    Q_k = Qs.sum(0)

    for itr in xrange(1,numpasses):
        # print "cvb0 iter", itr
        for i in xrange(doclen):
            Q_k -= Qs[i,:]
            Qs[i,:] = lik[i,:] * (Q_k + alpha)
            Qs[i,:] /= Qs[i,:].sum()
            Q_k += Qs[i,:]

    Q_k /= Q_k.sum()
    return Q_k

def predict(tokens, stopwords = False, alpha=1, numpasses=1, thresh1=1, thresh2=0.2):
    if len(tokens)>0:
        assert isinstance(tokens[0], unicode)
    invocab_tokens = [w.lower() for w in tokens if w.lower() in w2num]
    if stopwords:
        invocab_tokens = [word for word in invocab_tokens if word not in stop_words]

    # check that at least xx tokens are in vocabulary
    if len(invocab_tokens) < thresh1:
        return None  
    # check that at least yy% of tokens are in vocabulary
    elif len(invocab_tokens) / len(tokens) < thresh2:
        return None
    else:
        posterior = infer_cvb0(invocab_tokens, alpha=alpha, numpasses=numpasses)
        return posterior
