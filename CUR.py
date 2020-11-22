from sklearn.metrics import mean_squared_error
from math import sqrt
from svd import svd_matrix_construction
from svd import energy_90_percent
import time
import numpy as np

def spearman_corr(matrix,ultimate):
    sum=0
    cnt=0
    for i in range(0,len(matrix)):
        for j in range(0,len(matrix[i])):
            sum=sum+(matrix[i][j]-ultimate[i][j])**2
            cnt=cnt+1
    sum=6*sum
    hold=(cnt**3)-cnt
    value = 1 - (sum/hold)
    return value
    

def precision_on_K(matrix, ultimate):
    matrix_k = ultimate.tolist()
    cnt=0.00
    similar=0.00
    for i in range(0,len(matrix)):
        for j in range(0,len(matrix[i])):
            cnt=cnt+1
            a=int(round(matrix[i][j]))
            b=int(round(matrix_k[i][j]))
            if (a==b):
                similar = similar + 1
    precision=(similar*100)/cnt
    return precision/100


def CUR_decomposition(k):
    user_movie_matrix = np.load('train.npy')
    sum_of_squares = 0 
    
    user_length = user_movie_matrix.shape[0]
    movies_length = user_movie_matrix[0].size
    for i in range(user_length):
        for j in range(movies_length):
            sum_of_squares = sum_of_squares + user_movie_matrix[i][j]*user_movie_matrix[i][j]
    
    user_probability = []
    movie_probability = []
    
    for i in range(user_length):
        row_sum_of_sqaures = 0 
        for j in range(movies_length):
            row_sum_of_sqaures = row_sum_of_sqaures + user_movie_matrix[i][j]*user_movie_matrix[i][j]
        user_probability.append(row_sum_of_sqaures/sum_of_squares)
    
    for j in range(movies_length):
        column_sum_of_squares = 0
        for i in range(user_length):
            column_sum_of_squares = column_sum_of_squares + user_movie_matrix[i][j]*user_movie_matrix[i][j]
        movie_probability.append(column_sum_of_squares/sum_of_squares)
    
    selected_users = np.random.choice(len(user_probability),k, replace=False, p=user_probability) 
    selected_movies = np.random.choice(len(movie_probability),k, replace=False, p=movie_probability) 
    selected_movies.sort()
    selected_users.sort()
    
    C = []
    R = []
    for i in selected_users:
        R.append(list(user_movie_matrix[i]/sqrt(k*user_probability[i])))
    for j in selected_movies:
        C.append(list(user_movie_matrix[:,j]/sqrt(k*movie_probability[j])))
    Ct = np.transpose(C)
    
    W = []
    
    for i in selected_users:
        X=[]
        for j in selected_movies:
            X.append(user_movie_matrix[i][j])
        W.append(np.array(X))
    W = np.array(W)
    
    x,yt,sigma = svd_matrix_construction(W)
    pseudo_inverse_sigma = np.linalg.pinv(sigma) 
    square_pseudo = np.linalg.matrix_power(pseudo_inverse_sigma, 2)
    y = np.transpose(yt)
    xt = np.transpose(x)
    
    U = np.matmul(y, square_pseudo)
    U = np.matmul(U, xt)    
    np.save('CUR_Ct.npy', Ct)
    np.save('CUR_R.npy', R)
    
    new_x, new_yt, new_sigma = energy_90_percent(x,yt,sigma)
    pseudo_sigma_new = np.linalg.pinv(new_sigma)
    new_square_pseudo = np.linalg.matrix_power(pseudo_sigma_new, 2)
    y = np.transpose(new_yt)
    xt = np.transpose(new_x)
    
    U = np.matmul(y, new_square_pseudo)
    U = np.matmul(U, xt)
    np.save('CUR_U.npy', U)
    
def CUR_decomposition_90(k):
    user_movie_matrix = np.load('train.npy')
    sum_of_squares = 0 
    user_length = user_movie_matrix.shape[0]
    movies_length = user_movie_matrix[0].size
    
    for i in range(user_length):
        for j in range(movies_length):
            sum_of_squares = sum_of_squares + user_movie_matrix[i][j]*user_movie_matrix[i][j]
    user_probability = []
    movie_probability = []
    
    for i in range(user_length):
        row_sum_of_sqaures = 0 
        for j in range(movies_length):
            row_sum_of_sqaures = row_sum_of_sqaures + user_movie_matrix[i][j]*user_movie_matrix[i][j]
        user_probability.append(row_sum_of_sqaures/sum_of_squares)
    for j in range(movies_length):
        column_sum_of_squares = 0
        for i in range(user_length):
            column_sum_of_squares = column_sum_of_squares + user_movie_matrix[i][j]*user_movie_matrix[i][j]
        movie_probability.append(column_sum_of_squares/sum_of_squares)
    
    selected_users = np.random.choice(len(user_probability),k, replace=False, p=user_probability) 
    selected_movies = np.random.choice(len(movie_probability),k, replace=False, p=movie_probability)
    selected_movies.sort()
    selected_users.sort()
    
    C = []
    R = []
    for i in selected_users:
        R.append(list(user_movie_matrix[i]/sqrt(k*user_probability[i])))
    for j in selected_movies:
        C.append(list(user_movie_matrix[:,j]/sqrt(k*movie_probability[j])))
    Ct = np.transpose(C)
    W = []
    for i in selected_users:
        X=[]
        for j in selected_movies:
            X.append(user_movie_matrix[i][j])
        W.append(np.array(X))
    
    W = np.array(W)
    
    x,yt,sigma = svd_matrix_construction(W)
    pseudo_inverse_sigma = np.linalg.pinv(sigma)
    square_pseudo = np.linalg.matrix_power(pseudo_inverse_sigma, 2)
    y = np.transpose(yt)
    xt = np.transpose(x)
    U = np.matmul(y, square_pseudo)
    U = np.matmul(U, xt)
    
    np.save('CUR_90_Ct.npy', Ct)
    np.save('CUR_90_R.npy', R)

    new_x, new_yt, new_sigma = energy_90_percent(x,yt,sigma)
    pseudo_sigma_new = np.linalg.pinv(new_sigma)
    new_square_pseudo = np.linalg.matrix_power(pseudo_sigma_new, 2)
    y = np.transpose(new_yt)
    xt = np.transpose(new_x)
    
    U = np.matmul(y, new_square_pseudo)
    U = np.matmul(U, xt)
    np.save('CUR_90_U.npy', U) 
    

def main():
    
    t1=time.time()
    CUR_decomposition(1000)
    t2=time.time()
    
    
    t3 = time.time()
    CUR_decomposition_90(1000)
    t4=time.time()
    
    
    Ct = np.load('CUR_Ct.npy')
    A = np.load('train.npy')
    R = np.load('CUR_R.npy')
    U = np.load('CUR_U.npy')
    
    ultimate = np.matmul(Ct, U)
    ultimate = np.matmul(ultimate, R)
    RMSE_CUR=sqrt(mean_squared_error(A, ultimate))
    precisionTopK_CUR=precision_on_K(A, ultimate)
    SR_corr_CUR = spearman_corr(A, ultimate)
    
    Ct_90 = np.load('CUR_90_Ct.npy')
    R_90 = np.load('CUR_90_R.npy')
    U_90 = np.load('CUR_90_U.npy')   
    
    ultimate_90 = np.matmul(Ct_90, U_90)
    ultimate_90 = np.matmul(ultimate_90, R_90)
    RMSE_CUR_90=sqrt(mean_squared_error(A, ultimate_90))
    precisionTopK_CUR_90=precision_on_K(A, ultimate_90)
    SR_corr_CUR_90 = spearman_corr(A, ultimate_90)
    
    print('\n')
    print('CUR Result')
    print('RMSE = {}'.format(RMSE_CUR))
    print('Precision on top K = {}'.format(precisionTopK_CUR))
    print('Spearman Rank Correlation = {}'.format(SR_corr_CUR))
    print('Time Taken = {}'.format(t2-t1))
    
     
    print('\n')
    print('CUR 90% energy Result')
    print('RMSE = {}'.format(RMSE_CUR_90))
    print('Precision on top K = {}'.format(precisionTopK_CUR_90))
    print('Spearman Rank Correlation = {}'.format(SR_corr_CUR_90))
    print('Time Taken = {}'.format(t4-t3))

    
    del A
        
if __name__ == '__main__':
    main()
