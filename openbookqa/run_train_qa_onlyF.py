from models.bert_qa import BertQA
from models.bert_qa_cnn import BertQACNN

from tqdm import tqdm
import argparse
import pickle
import numpy as np
import os
import collections
import json
import operator
import torch
from random import shuffle
import gc
import random
gc.enable()

Input = collections.namedtuple("Input","idx passage a b c d label")

def read_ranked(fname,topk):
    fd = open(fname,"r").readlines()
    ranked={}
    for line in tqdm(fd,desc="Ranking "+fname+" :"):
        line = line.strip()
        out = json.loads(line)
        ranked[out["id"]]=out["ext_fact_global_ids"][0:topk]
    return ranked

def read_knowledge(fname):
    lines = open(fname,"r").readlines()
    knowledgemap = {}
    knowledge=[]
    for index,fact in tqdm(enumerate(lines),desc="Reading Knowledge:"):
        f=fact.strip().replace('"',"").lower()
        knowledgemap[f]=index
        knowledge.append(f)
    return knowledgemap,knowledge

# Returns itself if topk and train
def get_train_facts(fact1,topk,ranked,knowlegde,knowledgemap):
    if topk == 1:
        return [fact1]
    fset = []
    similarfacts = ranked[str(knowledgemap[fact1.lower()])]
    #print(similarfacts)
    for tup in similarfacts:
        f = knowledge[tup[0]]
        fset.append(f)
    if fact1.lower() in fset:
        return fset
    else :
        fset = fset[0:len(fset)-1]
        fset.append(fact1)
        assert len(fset) == topk
    return fset
    

def read_data_to_train(topk,ranked,knowledge,knowledgemap,use_gold_f2=False):
    data = {}
    gold = open("../data/hypothesis/hyp-gold-train.tsv","r").readlines()
    idx=0
    seq_lengths = []
    for line in tqdm(gold,desc="Preparing Train Dataset:"):
        line = line.strip().split("\t")
        qid = line[0]
        passage = line[1].split(" . ")
        choices = line[2:6]
        label = line[6]
        ans = choices[int(label)]
        fact1 = passage[0].strip()
        fact2 = passage[1].strip()
        if use_gold_f2:
            premise = line[1]
            data[idx]=Input(idx=idx,passage=premise,a=choices[0],b=choices[1],c=choices[2],d=choices[3],label=int(label))
        else:
            simfacts = get_train_facts(fact1,topk,ranked,knowledge,knowledgemap)
            for ix in range(0,1):
                shuffle(simfacts)
                premise = simfacts[0]
                for fact in simfacts[1:]:
                    premise = premise + " . " + fact
                data[idx]=Input(idx=idx,passage=premise,a=choices[0],b=choices[1],c=choices[2],d=choices[3],label=int(label))
                seq_lengths.append(len((premise + " " + choices[0]).split(" ")) + 3)
                seq_lengths.append(len((premise + " " + choices[1]).split(" ")) + 3)
                seq_lengths.append(len((premise + " " + choices[2]).split(" ")) + 3)
                seq_lengths.append(len((premise + " " + choices[3]).split(" ")) + 3)        
                idx+=1
                
    return data,np.percentile(seq_lengths,99)

def get_facts(facts,topk):
    return [tup[0] for tup in facts[0:topk]]

def get_test_facts(ranked_test,qid,topkf1,do_merge,knowledge):
    fact_map = {}
    merged_fact_map = {}
    for index in range(0,4):
        key = qid+"__ch_"+str(index)
        fact_p = ranked_test[key]
        facts = []
        for fact_index in fact_p:
            fact = knowledge[fact_index[0]]
            facts.append(fact)
            if fact not in merged_fact_map:
                merged_fact_map[fact]=fact_index[1]
            else:
                merged_fact_map[fact]+=fact_index[1]
            
        fact_map[index]=facts[0:topkf1]
    if not do_merge:
        return fact_map
    sorted_merged_facts = [tup[0] for tup in list(reversed(sorted(merged_fact_map.items(), key=operator.itemgetter(1))))[0:topkf1]]
    for index in range(0,4):
        fact_map[index]=sorted_merged_facts[0:topkf1]
    return fact_map
    
    
        
def read_data_to_test_unmerged(fname,lfname,ranked_test,knowledge,topkf1=0,topkf2=0,ftype="f1",do_merge=False,merge_f1=False,merge_f2=False):
    pickle_in = open(fname,"rb")
    rows = pickle.load(pickle_in)
    pickle_in.close()
    data = {}
    labeldict = {}
    seq_lengths = []
    for line in open(lfname).readlines():
        line = line.strip().split("\t")
        labeldict[line[0]] = line[-1]
            
    data = {}
    val = open(lfname,"r").readlines()
    for line in tqdm(val,desc="Preparing Test Dataset:"):
        line = line.strip().split("\t")
        qid = line[0]
        passage = line[1]
        choices = line[2:6]
        label = line[6]
        ans = choices[int(label)]
        fact1 = passage[0].strip()
        fact2 = passage[1].strip()
        
        facts_map = get_test_facts(ranked_test,qid,topkf1,do_merge,knowledge)

        for index,choice in enumerate(choices):
            qidx = qid + ":" + str(index)
            row = rows[qidx]
            facts = facts_map[index]
            premise = " . ".join(facts)    
            data[qidx]=Input(idx=qidx,passage=premise,a=choices[0],b=choices[1],c=choices[2],d=choices[3],label=int(label))
            seq_lengths.append(len((premise + " " + choices[0]).split(" ")) + 3)
            seq_lengths.append(len((premise + " " + choices[1]).split(" ")) + 3)
            seq_lengths.append(len((premise + " " + choices[2]).split(" ")) + 3)
            seq_lengths.append(len((premise + " " + choices[3]).split(" ")) + 3)


    return data,np.percentile(seq_lengths,99)
    
def print_qa_inputs(data,typ):
    print("Type :",typ)
    keys = list(data.keys())[0:5]
    for i in keys:
        print(data[i])

        
parser = argparse.ArgumentParser()

parser.add_argument("--exp",
                        default=None,
                        type=str,
                        required=False,
                        help="Experiment name")
parser.add_argument("--method",
                        default="sum",
                        type=str,
                        required=False,
                        help="Score Method, sum/max")
parser.add_argument("--max_seq",
                        default=128,
                        type=int,
                        required=False,
                        help="Sequence Length")
parser.add_argument("--topk", default=1, type=int, required=False,
                        help="TopK Facts to fetch")
parser.add_argument("--merged",
                        action='store_true',
                        help="Whether not to merge facts")
parser.add_argument("--gold",
                        action='store_true',
                        help="Use Gold Facts")
parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
parser.add_argument('--epochs',
                        type=int,
                        default=4,
                        help="random seed for initialization")

args = parser.parse_args()
    
exp = args.exp
topk = args.topk
max_seq = args.max_seq
seed = args.seed
epochs = args.epochs
method = args.method

is_merged = args.merged
use_gold_f2 = args.gold


knowledgemap,knowledge = read_knowledge("../data/knowledge/openbook.txt")

# f1topklist = [2,3,5]
# f1topklist = [2,3,4,5,6,7,8,]
# f2topklist = [2,3,4,5,6,7,8,]
f1topklist = [2,3,4,]
# f2topklist = [2,3,4,5]

# testdatadirs = ["../filterir/data/wordintersection/rankedf2-test-test-trained.pickled","../filterir/data/wordintersection/rankedf2-ir-top10-test-merged-intersect.pickled"]
# testdatadirs.extend(["../filterir/data/bagofwords/rankedf2-test-ir-not-merged-bow-test.pickled","../filterir/data/bagofwords/rankedf2-test-top10-merged-bow.pickled"])
# testdatadirs.extend(["../filterir/data/seq2seq/rankedf2-ir-test-trained-seq2seq.pickled","../filterir/data/seq2seq/rankedf2-ir-top10-merged-test-seq2seq.pickled"])

# testdatadirs = ["../filterir/data/wordintersection/rerankedf2-ir-test-test-trained.pickled","../filterir/data/wordintersection/rerankedf2-test-top10-merged-test.pickled"]
# testdatadirs.extend(["../filterir/data/bagofwords/rerankedf2-ir-test-ir-not-merged-bow-test.pickled","../filterir/data/bagofwords/rerankedf2-test-ir-top10-merged-bow-test.pickled"])
# testdatadirs.extend(["../filterir/data/seq2seq/rerankedf2-test-ir-test-trained-seq2seq.pickled","../filterir/data/seq2seq/rerankedf2-ir-top10-merged-test-seq2seq.pickled"])

# testdatadirs = ["../filterir/data/wordintersection/2rerankedf2-ir-test-test-trained.pickled","../filterir/data/wordintersection/2rerankedf2-test-top10-merged-test.pickled"]
testdatadirs = ["../filterir/data/wordintersection/2rerankedf2-ir-test-test-trained.pickled"]


basedir = "basemodel"

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

topkf2=0
for topkf1 in f1topklist:
    testdata_list = []
    test_max_seq_len = 0
    for fact_dir in ["../data/ranked/sts-trained-openbook.json","../data/ranked/cnn-openbook.json"]:
        topk = topkf1
        ranked_factfact = read_ranked("../data/ranked/sts-factfact.json",topk=topk)
        ranked_test = read_ranked(fact_dir,topk=topk)
        traindata,max_seq = read_data_to_train(topk,ranked_factfact,knowledge,knowledgemap)
        
        for testdir in testdatadirs:
            print("TestDir:"+testdir)
            testdata,test_max_seq = read_data_to_test_unmerged(testdir,"../data/hypothesis/hyp-gold-test.tsv",ranked_test,knowledge,topkf1,topkf2,ftype="f1",do_merge=False)
            test_max_seq_len = max(test_max_seq,test_max_seq_len)
            testdata_list.append(testdata)
            testdata,test_max_seq = read_data_to_test_unmerged(testdir,"../data/hypothesis/hyp-gold-test.tsv",ranked_test,knowledge,topkf1,topkf2,ftype="f1",do_merge=True,merge_f1=True)
            test_max_seq_len = max(test_max_seq,test_max_seq_len)
            testdata_list.append(testdata)
            
    max_seq = max(max_seq,test_max_seq)
    max_seq = int(max_seq)
    if max_seq < 128:
        max_seq = 128
        grad_acc_steps = 1
    elif max_seq < 256:
        max_seq = 256
        grad_acc_steps = 2
    elif max_seq < 394:
        max_seq = 394
        grad_acc_steps = 3
    else:
        max_seq = 512
        grad_acc_steps = 4

    output_dir = "/scratch/pbanerj6/fp16-bertqacnn-f1-"+basedir+"-"+str(topkf1)+str(topkf2)+"-"+str(max_seq)+"/"
    print("Output Directory:",output_dir)
    print_qa_inputs(traindata,"Train")
    # print_qa_inputs(valdata,"Val")
    for testdata in testdata_list:
        print_qa_inputs(testdata,"Test")
        
#     model =  BertQA( output_dir=output_dir,topk=topk,
#                  bert_model="bert-large-cased",do_lower_case=False,train_batch_size=32,seed=seed,learning_rate=1e-5,
#                  eval_batch_size=64,max_seq_length=max_seq,num_labels=4,grad_acc_steps=grad_acc_steps,
#                  num_of_epochs=epochs,action="train",fp16=True)
    
    model =  BertQACNN( output_dir=output_dir,topk=topk,
                 bert_model="bert-large-cased",do_lower_case=False,train_batch_size=32,seed=seed,learning_rate=1e-5,
                 eval_batch_size=64,max_seq_length=max_seq,num_labels=4,grad_acc_steps=2,
                 num_of_epochs=epochs,action="train",fp16=False)
    
    data = { "train":traindata, "test":testdata_list}
    _,metrics = model.train(data)
    outfd = open(output_dir+"results.txt","w+")
    outfd.write(testdir+"\n")
    outfd.write(json.dumps(metrics)+"\n")
    outfd.close()
    del model
    gc.collect