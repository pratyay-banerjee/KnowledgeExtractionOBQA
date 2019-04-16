# SML-575
## Evaluating Information Retrieval Models using BERT features and OpenBookQA 

# [NEW2]
1. Find the PCA reduced data for "all" data. Those who are having trouble running use this data to run your "all" model. Dimension of data (75000 x 6000) instead of (75000 x 23000).  
Data retaintion is 90%.  
Location : /scratch/kkpal/sml-dataset/
2. Find the pipeline of using PCA reduced code from svm/SVM-PCA-All-Linear.ipynb

# [NEW]
1. Added Precision@1 and Precision@3 metric code in Xgboost-All / Xgboost-Tokens. Kindly replicate those also with MRR for Test.
2. SVM trainSampleSize change accordingly. TrainingData maxSize : 75000.


Dataset is present in : /scratch/pbanerj6/sml-dataset/ranking 
 
1. train.tsv 
2. val.tsv : Use this for model selection
3. test.tsv : Use this only for measuring accuracy and mrr

Embedding files contain tokens and their corresponding embeddings of size 768. 

Tasks:
1. Evaluate classifiers using only [CLS] tokens.
2. Evaluate using all token embeddings.
3. Evaluate using Text tokens.
4. Create following table:

| Classifier | Text Tokens | CLS Emb | All Tokens| All PCA Tokens |
|------------|-------------|---------|-----------|-----------|
| LinearSVC  | Val: Test: MRR:| Val: Test: MRR: | Val: Test: MRR:| Val: Test: MRR:|
| SVM Poly  | Val: Test: MRR:| Val: Test: MRR: | Val: Test: MRR:| Val: Test: MRR:|
| SVM RBF  | Val: Test: MRR:| Val: Test: MRR: | Val: Test: MRR:|  Val: Test: MRR:|
 DecisionTree  | Val:56.18% Test:55.73% MRR,PR@1,PR@3:(0.435964502164501, 0.15, 0.5393333333333333)| Val:57.66% Test:56.80% MRR,PR@1,PR@3:(0.45552587782587617, 0.16666666666666666, 0.5866666666666667) | Val:55.84% Test:55.68% MRR,PR@1,PR@3:(0.4611036075036066, 0.18333333333333332, 0.5826666666666667)| Val: Test: MRR:|
| ExtraTrees  | Val:53.25% Test:52.59% MRR,PR@1,PR@3:(0.39411875901875804, 0.11933333333333333, 0.4866666666666667) | Val:55.36% Test:55.16% MRR,PR@1,PR@3:(0.44610519480519406, 0.16466666666666666, 0.57) | Val:54.39% Test:53.60% MRR,PR@1,PR@3:(0.40922029822029676, 0.126, 0.4573333333333333)| Val: Test: MRR:|
| RandomForest | Val:58.65% Test:57.00% MRR,PR@1,PR@3:(0.4768623857623865, 0.222, 0.602)| Val:59.03% Test:59.21% MRR,PR@1,PR@3:(0.5292222222222241, 0.2906666666666667, 0.6973333333333334) | Val:58.76% Test:57.88% MRR,PR@1,PR@3:(0.5313962962962976, 0.29533333333333334, 0.69)| Val: Test: MRR:|
| ExtraTreesEnsemble | Val:57.53% Test:57.06% MRR,PR@1,PR@3:(0.4838741221741232, 0.24066666666666667, 0.6126666666666667)| Val:58.50% Test:58.32% MRR,PR@1,PR@3:(0.5193444444444472, 0.2826666666666667, 0.6713333333333333) | Val:57.74% Test:56.74% MRR,PR@1,PR@3:(0.5080313131313139, 0.26866666666666666, 0.6526666666666666)| Val: Test: MRR:|


5. Plot learning curves for : 100,1000,10000,20000,50000 training samples and all test samples, test accuracy and mrr.
6. Save your best models.




Tips to work on Jupyter in Agave:

1. Start an interactive session : interactive -n 20 
2. Note the host you start your session : "cg6-17"
3. source activate your_conda_env
4. pip install jupyterlab
5. unset XDG_RUNTIME_DIR
6. To start jupyter : jupyter lab --port=8889 --no-browser --ip=0.0.0.0 
Note the port 8889
7. You can run jupyter in nohup like : nohup jupyter lab --port=8889 --no-browser --ip=0.0.0.0 > jupyter.log &
8. Note the token generated in jupyter.log 
9. In another terminal/cmd prompt ssh to agave using port forwarding : ssh -L8889:localhost:8889 yourname@agave.asu.edu
10. In this new terminal/cmd prompt ssh to "cg6-17" using port forwarding : ssh -L8889:localhost:8889 yourname@cg6-17
11. In your browser open : localhost:8889. If a token is asked enter the token from jupyter.log




