# KnowledgeExtractionOBQA
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
| DecisionTree  | Val:56.18% Test:55.73% MRR,PR@1,PR@3:(0.435964502164501, 0.15, 0.5393333333333333)| Val:57.66% Test:56.80% MRR,PR@1,PR@3:(0.45552587782587617, 0.16666666666666666, 0.5866666666666667) | Val:55.84% Test:55.68% MRR,PR@1,PR@3:(0.4611036075036066, 0.18333333333333332, 0.5826666666666667)| Val:55.64% Test:55.93% MRR,PR@1,PR@3:(0.4412522366522349, 0.14466666666666667, 0.5573333333333333)|
| ExtraTrees  | Val:53.25% Test:52.59% MRR,PR@1,PR@3:(0.39411875901875804, 0.11933333333333333, 0.4866666666666667) | Val:55.36% Test:55.16% MRR,PR@1,PR@3:(0.44610519480519406, 0.16466666666666666, 0.57) | Val:54.39% Test:53.60% MRR,PR@1,PR@3:(0.40922029822029676, 0.126, 0.4573333333333333)| Val:51.01% Test:50.25% MRR,PR@1,PR@3:(0.3928292448292439, 0.13733333333333334, 0.44)|
| RandomForest | Val:58.65% Test:57.00% MRR,PR@1,PR@3:(0.4768623857623865, 0.222, 0.602)| Val:59.03% Test:59.21% MRR,PR@1,PR@3:(0.5292222222222241, 0.2906666666666667, 0.6973333333333334) | Val:58.76% Test:57.88% MRR,PR@1,PR@3:(0.5313962962962976, 0.29533333333333334, 0.69)| Val:50.53% Test:50.21% MRR,PR@1,PR@3:(0.40951861471861456, 0.16133333333333333, 0.506)| Val:52.59% Test:52.24% MRR,PR@1,PR@3:0.21 sec|
| ExtraTreesEnsemble | Val:57.53% Test:57.06% MRR,PR@1,PR@3:(0.4838741221741232, 0.24066666666666667, 0.6126666666666667)| Val:58.50% Test:58.32% MRR,PR@1,PR@3:(0.5193444444444472, 0.2826666666666667, 0.6713333333333333) | Val:57.74% Test:56.74% MRR,PR@1,PR@3:(0.5080313131313139, 0.26866666666666666, 0.6526666666666666)| Val:50.53% Test:50.21% MRR,PR@1,PR@3:(0.40951861471861456, 0.16133333333333333, 0.506)|
| GaussianNB | Val:58.15% Test:57.27% MRR,PR@1,PR@3:(0.4506448773448774, 0.124, 0.6606666666666666) | Val:50.00% Test:50.00% MRR,PR@1,PR@3:(0.2857142857142808, 0.0, 0.0) | Val:50.00% Test:50.00% MRR,PR@1,PR@3:(0.2857142857142808, 0.0, 0.0) | Val:50.00% Test:50.00% MRR,PR@1,PR@3:(0.2857142857142808, 0.0, 0.0) |
| MultinomialNB | Val:58.17% Test:58.07% MRR,PR@1,PR@3:(0.5148333333333348, 0.27666666666666667, 0.6753333333333333) | Val:58.09% Test:57.85% MRR,PR@1,PR@3:(0.5344666666666683, 0.294, 0.7026666666666667) | Val:54.79% Test:54.29% MRR,PR@1,PR@3:(0.32872857142856615, 0.002, 0.44533333333333336) | Val:54.62% Test:54.49% MRR,PR@1,PR@3:(0.4761777777777788, 0.21133333333333335, 0.6453333333333333) |
| BernoulliNB | Val:58.81% Test:58.79% MRR,PR@1,PR@3:(0.5170132756132769, 0.2813333333333333, 0.6653333333333333) | Val:50.00% Test:50.00% MRR,PR@1,PR@3:(0.4318788359788366, 0.18466666666666667, 0.5533333333333333) | Val:50.00% Test:50.00% MRR,PR@1,PR@3:(0.2857142857142808, 0.0, 0.0) | Val:50.00% Test:50.00% MRR,PR@1,PR@3:(0.41022592592592627, 0.168, 0.508) |
| Logistic Regression | Val:60.31% Test:59.36% MRR,PR@1,PR@3:(0.5208888888888904, 0.2806666666666667, 0.6786666666666666)| Val:64.72% Test:65.11% MRR,PR@1,PR@3:(0.6008666666666689, 0.376, 0.7773333333333333)| Val:59.56% Test:58.38% MRR,PR@1,PR@3:(0.5391238095238106, 0.312, 0.686)| Val: Test: MRR,PR@1,PR@3:|
| Passive Aggressive Classifier | Val:60.63% Test:59.17% MRR,PR@1,PR@3:(0.5160222222222234, 0.2786666666666667, 0.6646666666666666)| Val:63.93% Test:64.25% MRR,PR@1,PR@3:(0.5971000000000024, 0.37133333333333335, 0.7813333333333333) | Val: Test: MRR,PR@1,PR@3:| Val:63.02% Test:62.95% MRR,PR@1,PR@3:(0.5707571428571455, 0.34933333333333333, 0.7313333333333333)|
| Perceptron | Val:58.71% Test:57.29% MRR,PR@1,PR@3:(0.48942380952381076, 0.25133333333333335, 0.6253333333333333)| Val:64.29% Test:65.07% MRR,PR@1,PR@3:(0.5967888888888915, 0.37133333333333335, 0.778)| Val: Test: MRR,PR@1,PR@3:| Val:61.00% Test:61.11% MRR,PR@1,PR@3:(0.5557571428571451, 0.33, 0.712)|
| SGD Classifier | Val:60.69% Test:59.39% MRR,PR@1,PR@3:(0.5208333333333353, 0.2793333333333333, 0.6833333333333333)| Val:64.69% Test:64.27% MRR,PR@1,PR@3:(0.6009222222222241, 0.37666666666666665, 0.7733333333333333) | Val: Test: MRR,PR@1,PR@3:| Val:64.29% Test:64.53% MRR,PR@1,PR@3:(0.5971925925925942, 0.37666666666666665, 0.7766666666666666)|


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




