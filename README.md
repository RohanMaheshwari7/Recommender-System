# Recommender-System

This project is aimed at implementing and comparing various techniques for building a Recommender System for the given dataset. 
Several techniques were used to achieve this
a. Collaborative filtering 
b. SVD 
c. CUR 
For each method used, RMSE, Precision on top K, Spearman Rank Correlation and time taken for computation was calculated and based on that comparison could be made on the methods used. 

## Team Members ##
- Soumil Agarwal &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   2017B4A71606H
- Giridhar Bajpai  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2017B4A71451H
- Rohan Maheshwari &nbsp;&nbsp;  2017B4A70965H


## Working: ##
The data was extracted from https://grouplens.org/datasets/movielens/. The code should be run in the following order 
1. train_test_split.py
2. dataprocessing.py

The above two files are used for preprocessing. After running the above two files, the following files can be run in any order to obtain results of the required techinique. 
1. Collaboartive Filtering : cf.py
2. Singular Value Decomposition: svd.py
3. CUR Decomposition : CUR.py
