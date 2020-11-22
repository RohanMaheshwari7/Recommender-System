import time
import pandas as pd
import numpy as np
import colabfilter
import svd
import cur

def main():
	data = pd.read_csv('ratings.dat',sep="::",usecols=[0,1,2],names=['user','movie','rating'],engine='python')
	# data = data.drop(['timestamp'],axis = 1)
	print(data.head())
	print(data.describe()) #6040 users x 3952 movies

	A = np.zeros((6041,3953))
	for i in range(len(data)):
		A[data.loc[i,"user"]][data.loc[i,"movie"]] = data.loc[i,"rating"]
	
	print("Matrix created successfully")
	#do collaborative filtering 


	#do svd


	#do cur

if __name__ == "__main__":
	main()
