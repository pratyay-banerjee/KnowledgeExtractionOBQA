
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import joblib
import time

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def load_cls(fname):
    return np.array(pickle.load(open(fname,"rb")))


# In[3]:


path = "/scratch/pbanerj6/sml-dataset/"
trainSamples = 75000


# In[5]:


X_train = load_cls(path+"X_train_tokens.p")[:trainSamples]
X_val_cls = load_cls(path+"X_val_tokens.p")


# In[6]:


y_train = load_cls(path+"y_train.p")[:trainSamples]
y_val = load_cls(path+"y_val.p")


# In[7]:


#X_train[0]


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer

def my_preprocessor(doc):
    return doc

# tokenize the doc and lemmatize its tokens
def my_tokenizer(doc):
    return doc

custom_vec = CountVectorizer(preprocessor=my_preprocessor, tokenizer=my_tokenizer)
cwm = custom_vec.fit_transform(X_train)
tokens = custom_vec.get_feature_names()

print("Feature extraction Done")
# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm


# In[10]:


feature_pipeline = Pipeline([
('vect',  CountVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3),preprocessor=my_preprocessor, tokenizer=my_tokenizer)),
('tfidf', TfidfTransformer()),
])
print ("Feature pipeline done")

# In[11]:


X_train_f = feature_pipeline.fit_transform(X_train)
X_val_f =feature_pipeline.transform(X_val_cls)

print("Feature transformation Done")
# In[12]:


X_train_f[0]


# In[13]:


model = svm.SVC(gamma='scale',verbose=True,probability=True,cache_size=7000,max_iter=30000)
stime = time.time()
model.fit(X_train_f,y_train)
print("Training Time:",time.time()-stime)


# In[14]:


from sklearn.metrics import accuracy_score
stime=time.time()
preds_val = model.predict(X_val_f)
accuracy = accuracy_score(y_val, preds_val)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Validation Time:",time.time()-stime)


# In[15]:


X_test = load_cls(path+"X_test_tokens.p")
y_test = load_cls(path+"y_test.p")


# In[16]:


X_test_f =feature_pipeline.transform(X_test)


# In[17]:


stime=time.time()
preds_test = model.predict(X_test_f)
accuracy = accuracy_score(y_test, preds_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Test Time:",time.time()-stime)


# In[18]:


from scipy.stats import rankdata

def mrrs(out, labels):
#     print(out,labels)
    outputs = np.argmax(out,axis=1)
    mrr = 0.0 
    for label,ranks in zip(labels,out):
        ranks = rankdata(ranks*-1)
        rank = ranks[label]
#         print(rank,ranks)
        mrr+=1/rank
    return mrr/len(labels)

def mrrwrapper(qid2c,qid2indexmap,preds_prob):
    labels = []
    out = []
    for qid in qid2c.keys():
        scores = []
        for ix in qid2indexmap[qid]:
            if len(scores) < 6:
                scores.append(preds_prob[ix][1])
        if len(scores) < 6:
            continue
        out.append(scores)
        labels.append(int(qid2c[qid]))
    return mrrs(np.array(out),labels)

def load_ranking(fname):
    return pickle.load(open(path+"ranking_"+fname+".p","rb"))


# In[19]:


preds_test_probs = model.predict_proba(X_test_f)


# In[20]:


qid2c,qid2indexmap = load_ranking("test")


# In[21]:


print("MRR:",mrrwrapper(qid2c,qid2indexmap,preds_test_probs))

