import math
import time
import pandas as pd
from sklearn.model_selection import train_test_split

def createUtilityMatrix(df):
	'''
	Helper function to create user x movie matrix and other utilities 
	'''
	matrix={}
	inverse_matrix={}
	l = len(df)
	# Bias for each user
	biasU={}
	countU={}
	# Bias for each movie
	biasM={}
	countM={}

	rating_sum = 0

	for i in df.index:
		user = df['user'][i]
		movie = df['movie'][i]
		rating = df['rating'][i]
		rating_sum += rating

		if matrix.get(user):
			matrix[user][movie] = rating
		else:
			matrix[user] = {movie: rating}
		if inverse_matrix.get(movie):
			inverse_matrix[movie][user] = rating
		else:
			inverse_matrix[movie] = {user: rating}

		if biasU.get(user):
			biasU[user] += rating
			countU[user] += 1
		else:
			biasU[user] = rating
			countU[user] = 1

		if biasM.get(movie):
			biasM[movie] += rating
			countM[movie] += 1
		else:
			biasM[movie] = rating
			countM[movie] = 1

	global_mean = rating_sum/l
	for user in biasU:
			biasU[user] = biasU[user] / countU[user]
			biasU[user] = biasU[user] - global_mean
	for movie in biasM:
		biasM[movie] = biasM[movie] / countM[movie]
		biasM[movie] = biasM[movie] - global_mean

	return matrix, inverse_matrix, biasU, biasM, global_mean

def predict_item(A_train, inv_mat, biasU, biasM, gm, user, movie):
	'''
	Uses item - item filtering approach
	Predicts the rating given by user 'user' to movie 'movie'
    Parameters
    ----------
    A_train : dict
    	utility matrix in dictionary format
    inv_mat : dict
    	inverse dictionary from movies to users
    biasU : dict
		User bias
    biasM : dict
		Movie bias
    gm : integer
		global mean over all the movies
    user : integer
		userID
    movie : integer
    	movieID
    Returns
    ----------
    rating : integer
    	predicted rating
	'''
	ium = inv_mat
	um = A_train
	mu = gm
	if um.get(user) == None:
		if ium.get(movie) == None:
			return mu
		else:
			return biasM[movie] + mu
	elif ium.get(movie) == None:
		return mu + biasU[user]
	ix = ium[movie]
	b1 = -(biasM[movie] + mu)
	scores = []
	for moviey in um[user]:
		if moviey == movie:
			continue
		iy = ium[moviey]
		b2 = -(biasM[moviey] + mu)
		sxy = sim(ix, iy, b1, b2)
		scores.append((sxy, um[user][moviey] + b2))
	scores.sort(reverse=True)
	return get_rating(scores, -b1)

def predict_user(A_train, inv_mat, biasU, biasM, gm, user, movie):
	'''
	Uses user - user filtering to find predicted rating
	'''
	ium = inv_mat
	um = A_train
	mu = gm
	if um.get(user) == None:
		if ium.get(movie) == None:
			return mu
		else:
			return biasM[movie] + mu
	elif ium.get(movie) == None:
		return mu + biasU[user]
	ix = um[user]
	b1 = -(biasU[user] + mu)
	scores = []
	for usery in ium[movie]:
		if usery == user:
			continue
		iy = um[usery]
		b2 = -(biasU[usery] + mu)
		sxy = sim(ix, iy, b1, b2)
		scores.append((sxy, ium[movie][usery] + b2))
	scores.sort(reverse=True)
	return get_rating(scores, -b1)

def predict_baseline(A_train, inv_mat, biasU, biasM, gm, user, movie):
	'''
	Uses baseline approach in collaborative filtering to find predicted rating
	'''
	ium = inv_mat
	um = A_train
	mu = gm
	if um.get(user) == None:
		if ium.get(movie) == None:
			return mu
		else:
			return biasM[movie] + mu
	elif ium.get(movie) == None:
		return mu + biasU[user]
	baseline = mu + biasU[user] + biasM[movie]
	ix = ium[movie]
	b1 = -(biasM[movie] + mu)
	scores = []
	for moviey in um[user]:
		if moviey == movie:
			continue
		iy = ium[moviey]
		b2 = -(biasM[moviey] + mu)
		sxy = sim(ix, iy, b1, b2)
		baseliney = mu + biasU[user] + biasM[moviey]
		scores.append((sxy, um[user][moviey] - baseliney))
	scores.sort(reverse=True)
	return get_rating(scores, baseline)

def get_rating(scores, base):
	'''
	Computes the rating using weighted average of top k scores
	'''
	l = min(5, len(scores))
	rating = 0
	den = 0
	for i in range(l):
		rating += (scores[i][0]) * scores[i][1]
		den += abs(scores[i][0])
	if den == 0:
		return bound(base)
	rating = rating / den
	rating += base
	return bound(rating)

def sim(vec1, vec2, b1, b2):
	'''
	Finds cosine similarity between given two vectors
	Arguments:
		vec1: first vector
		vec2: second vector
		b1: bias for first vector
		b2: bias for second vector
	'''
	b1 = 0
	b2 = 0
	sim = 0
	# Normalization constants
	n1 = 0
	n2 = 0
	for feature in vec1:
		a = vec1[feature] + b1
		if vec2.get(feature):
			b = vec2[feature] + b2
			sim += a * b
		n1 += a * a
	for feature in vec2:
		b = vec2[feature] + b2
		n2 += b * b
	if sim == 0:
		return 0
	n1 = math.sqrt(n1)
	n2 = math.sqrt(n2)
	return sim / (n1 * n2)

def bound(rating):
	'''
	Bounds the rating in the range [1, 5]
	'''
	return min(max(rating, 1), 5)

def calc_loss(data):
	'''
	Computes RMSE loss, Spearman rank correlation and Precision at rank k for the data given in utilmat
	'''
	data = data.sample(frac = 1)
	split_value = 100
	train_data = data.iloc[:-split_value, :]
	test_data = data.iloc[-split_value:,:]
	# print(test_data.head())

	A_train, inv_mat, biasU, biasM, gm = createUtilityMatrix(train_data)
	A_test, inv_mat_test, biasU_test, biasM_test, gm_test = createUtilityMatrix(test_data)

	# print("Matrices created successfully")

	testmatrix = A_test
	#Root mean square 
	rmse_u = 0
	rmse_i = 0
	rmse_b = 0
	#Spearman Rank Correlation
	src_u = 0
	src_i = 0
	src_b = 0
	cnt = 0
	fullcnt = 0
	#precision at rank 5
	precision_k_items_vals = []
	precision_k_baseline_vals = []
	start = time.time()
	for user in testmatrix:
		#dict containing pred->actual
		precision_k_item = {}
		precision_k_baseline = {}
		for movie in testmatrix[user]:
			actual = testmatrix[user][movie]
			# predu = predict_user(A_train, inv_mat, biasU, biasM, gm, user, movie)
			predu = 0
			predi = predict_item(A_train, inv_mat, biasU, biasM, gm, user, movie)
			predb = predict_baseline(A_train, inv_mat, biasU, biasM, gm, user, movie)
			precision_k_item[predi] = actual
			precision_k_baseline[predb] = actual
			rmse_u += (actual - predu) ** 2
			rmse_i += (actual - predi) ** 2
			rmse_b += (actual - predb) ** 2
			# print(cnt,user,movie,actual,predi,predb)
			cnt += 1
		#sort in reverse based on first val and check first 5
		pcnt = 0
		num = 0
		den = 0
		for p in sorted(precision_k_item,reverse=True):
			if p > 3.5:
				if precision_k_item[p] > 3.5:
					num+=1
				den+=1
			else:
				break
			pcnt+=1
			if pcnt==5:
				break
		if den>0:
			precision_k_items_vals.append((num/den))

		pcnt = 0
		num = 0
		den = 0
		for p in sorted(precision_k_baseline,reverse=True):
			if p > 3.5:
				if precision_k_baseline[p] > 3.5:
					num+=1
				den+=1
			else:
				break
			pcnt+=1
			if pcnt==5:
				break
		if den>0:		
			precision_k_baseline_vals.append((num/den))

	src_u = 1 - ((6 * rmse_u)/(cnt ** 3 - cnt))
	src_i = 1 - ((6 * rmse_i)/(cnt ** 3 - cnt))
	src_b = 1 - ((6 * rmse_b)/(cnt ** 3 - cnt))
	rmse_b /= cnt
	rmse_u /= cnt
	rmse_i /= cnt

	precision_i = 0
	precision_b = 0
	for i in range(len(precision_k_items_vals)):
		precision_i += precision_k_items_vals[i]
	precision_i /= len(precision_k_items_vals)

	for i in range(len(precision_k_baseline_vals)):
		precision_b += precision_k_baseline_vals[i]
	precision_b /= len(precision_k_baseline_vals)
	print("Time taken for collaborative filtering = ",time.time() - start)
	return rmse_b, rmse_i, src_b, src_i, precision_b, precision_i 