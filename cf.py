import time
import pandas as pd
import numpy as np
import colabfilter


def main():
	data = pd.read_csv('ratings.dat',sep="::",usecols=[0,1,2],names=['user','movie','rating'],engine='python')
	# data = data.drop(['timestamp'],axis = 1)
	print(data.head())
	# print(data.describe()) #6040 users x 3952 movies


	
	#do collaborative filtering 
	rmse_b, rmse_i, src_b, src_i, precision_b, precision_i = colabfilter.calc_loss(data)
	print("RMSE values are: ")
	print('Using item-item filtering: ', rmse_i)
	print('Using baseline approach: ', rmse_b)
	# print('Using user-user filtering: ', rmse_u)
	

	print("Spearman Rank Correlation values are: ")
	print('Using item-item filtering: ', src_i)
	print('Using baseline approach: ', src_b)
	# print('Using user-user filtering: ', src_u)
	

	print("Precision at rank 5 are: ")
	print('Using item-item filtering: ', precision_i)
	print('Using baseline approach: ', precision_b)

if __name__ == "__main__":
	main()
