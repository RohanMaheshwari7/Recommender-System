import time
import pandas as pd
import colabfilter
import svd
import cur

def main():
	data = pd.read_csv('ratings.csv')
	data = data.drop(['timestamp'],axis = 1)
	print(data.head())
	#do collaborative filtering 


	#do svd


	#do cur

if __name__ == "__main__":
	main()
