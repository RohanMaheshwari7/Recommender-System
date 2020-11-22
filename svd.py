import numpy as np
from math import sqrt
import time


def compuation(train_new, testData):
    """
    Parameters
    ----------
    train_new : numpy.ndarray
        Training Set values
    testData : numpy.ndarray
        Test Set Values
    Returns
    -------
    RMSE : float
        Returns the Root Mean Square Error value
    SR_corr : float
        Returns the Spearman Rank Correlation
    top_K_precision : float
        Return the precision on Top K value
    """
    users_length = len(train_new)
    item_length = len(train_new[0])
    error_square, test_n = 0, 0
    for user in range(users_length):
        for item in range(item_length):
            if testData[user, item] == 0:
                continue
            else:
                error_square += (testData[user, item] - train_new[user, item])**2
                test_n += 1
    
    SR_corr = 1 - ((6 * error_square) / (test_n * (test_n**2 - 1)))
    RMSE = (error_square / test_n) ** 0.5

    top_K_precision = 0
    K = 10
    THRESHOLD = 3.5
    rated_movies_count = {}
    i = 0
    for user in testData:
        rated_movies_count[i] = user[user > 0].size
        i = i + 1
    i = 0
    precision_values = []
    for user in train_new:
        if rated_movies_count[i] < K:
            i = i + 1
            continue
        top_k_indices = (-user).argsort()[:K]
        top_k_values = [(index, user[index]) for index in top_k_indices]
        recommendations = []
        for (index, user[index]) in top_k_values:
            if(user[index] >= 3.5):
                recommendations.append((index, user[index]))
                if len(recommendations) == K:
                    break
        cnt = 0
        for tup in recommendations:
            if testData[i][tup[0]] >= THRESHOLD:
                cnt = cnt + 1
        if len(recommendations) > 0:
            precision = cnt/len(recommendations)
            precision_values.append(precision)
        i = i + 1
    top_K_precision = sum(precision_values) / len(precision_values)
    return RMSE, SR_corr, top_K_precision


def corr_matrix_construction(data, file_name):
    """
    

    Parameters
    ----------
    data : int
        user count
    file_name : string
        File passed with previously obtained values

    Returns
    -------
    matrix_correlation : np.ndarray
        Correlation matrix obtained

    """
    users_length = len(data)
    matrix_correlation = np.corrcoef(data)[:users_length+1, :users_length+1]
    np.save(file_name, matrix_correlation)
    return matrix_correlation

def collaborative_basic(trainData, testData, matrix_correlation, K):
    """
    

    Parameters
    ----------
    trainData : np.ndarray
        Training dataset values
    testData : np,ndarray
        Test dataset values
    matrix_correlation : np.ndarray
        correlation matrix
    K : int
        

    Returns
    -------
    train_new : np.ndarray
        Reconstructed training set

    """
    users_length = len(trainData)#6040
    item_length = len(trainData[0])#3952
    
    train_new = np.zeros((users_length, item_length))
    for predicted_users in range(users_length):
        K_closest = (-matrix_correlation[predicted_users]).argsort()[:K]
        for item in range(item_length):
            if testData[predicted_users, item] == 0:
                continue
            sum_corr = 0
            for user_close in K_closest:
                if trainData[user_close, item] != 0:
                    train_new[predicted_users, item] += matrix_correlation[predicted_users, user_close] * trainData[user_close, item]
                    sum_corr += matrix_correlation[predicted_users, user_close]
            if sum_corr != 0:
                train_new[predicted_users, item] /= sum_corr
    return train_new


def svd_matrix_construction(A):
    """
    Function for SVD to get U , sigma and Vt
    
    Parameters
    ----------
    A : np.ndarray
        Utility Matrix
    Returns
    -------
    u : np.ndarray
        user to conept similarlity matrix
    vt : np.ndarray
        transpose of movie to concept similarity matrix
    sigma : np.ndarray
        strength of each concept
    Normalizes the Utility matrix consisting of users, movies and their ratings by
    replacing 0s in a row by their row mean.Performs SVD on the normalized utility
    matrix and factorizes it into U, S and V*
    
    """
    
    At = np.transpose(A)
    A_At = np.matmul(A, At)
    num_users = A.shape[0]
    num_movies = A.shape[1]
    At_A = np.matmul(At, A)
    del A
    del At
    
    eigen_value_u, eigen_vector_u = np.linalg.eigh(A_At)
    eigen_value_v, eigen_vector_v = np.linalg.eigh(At_A)
    
    pos_eigen_u = []
    pos_eigen_v = []
    
    for val in eigen_value_u.tolist():
        if(val > 0):
            pos_eigen_u.append(val)
    for val in eigen_value_v.tolist():
        if(val > 0):
            pos_eigen_v.append(val)
    
    pos_eigen_u.reverse()
    pos_eigen_v.reverse()
    squareroot_eigen_u = [sqrt(val) for val in pos_eigen_u]
    squareroot_eigen_u = np.array(squareroot_eigen_u)
    
    sigma = np.diag(squareroot_eigen_u)
    sigma_size = sigma.shape[0]
    
    ut = np.zeros(shape = (sigma_size, num_users))
    vt = np.zeros(shape = (sigma_size, num_movies))
    i = 0
    for val in pos_eigen_u:
        ut[i] = eigen_vector_u[eigen_value_u.tolist().index(val)]
        i = i + 1
    i = 0
    for val in pos_eigen_v:
        vt[i] = eigen_vector_v[eigen_value_v.tolist().index(val)]
        i = i + 1
    u = np.transpose(ut)
    del ut
    return u, vt, sigma

def energy_90_percent(u, vt, sigma):
    """
    Function for SVD with 90% retained energy    
    Parameters
    ----------
    u : np.ndarray
        user to conept similarlity matrix
    vt : np.ndarray
        transpose of movie to concept similarity matrix
    sigma : np.ndarray
        strength of each concept
    Returns
    -------
    u_n : np.ndarray
        Modified u
    vt_n : np.ndarray
        Modied vt
    sigma_n : np.ndarray
        Modied sigma
    """
    
    sigma_size = sigma.shape[0]
    sigma_square_sum = 0
    eigen_values_reqd = np.zeros(sigma_size)
    
    for i in range(sigma_size):
        sigma_square_sum += sigma[i][i] * sigma[i][i]
    curr_sum = 0
    
    for i in range(sigma_size):
        curr_sum += sigma[i][i] * sigma[i][i]
        eigen_values_reqd[i] = sigma[i][i]
        if (curr_sum/sigma_square_sum) >= 0.9:
            i = i + 1
            break
    
    eigen_values_reqd = eigen_values_reqd[eigen_values_reqd > 0]
    sigma_n = np.diag(eigen_values_reqd)
    u_n = np.transpose(np.transpose(u)[:sigma_n.shape[0]])
    vt_n = vt[:sigma_n.shape[0]]
    return u_n, vt_n, sigma_n


def main():
    
    K = 50
    trainData = np.load('train.npy')
    testData = np.load('test.npy')
    
    
    try:
        svd_matrix_correlation = np.load('svd_correlation_matrix.npy')
        svd_90_matrix_correlation = np.load('svd_90_correlation_matrix.npy')
    except FileNotFoundError:
        u, vt, sigma = svd_matrix_construction(trainData)
        u_n, vt_n, sigma_n = energy_90_percent(u, vt, sigma)
        user_projection = np.matmul(u, sigma)
        users_projection_90 = np.matmul(u_n, sigma_n)
        
        svd_matrix_correlation = corr_matrix_construction(user_projection, 'svd_correlation_matrix.npy')
        svd_90_matrix_correlation = corr_matrix_construction(users_projection_90, 'svd_90_correlation_matrix.npy')
    
    
    t0 = time.time()
    
    result_svd = collaborative_basic(trainData, testData, svd_matrix_correlation, K)
    RMSE_svd, SR_corr_svd, precisionTopK_svd = compuation(result_svd, testData)
    
    #del result_svd
    #del svd_matrix_correlation
    
    t1 = time.time()
    print('\n')
    print('SVD Result')
    print('RMSE = {}'.format(RMSE_svd))
    print('Precision on top K = {}'.format(precisionTopK_svd))
    print('Spearman Rank Correlation = {}'.format(SR_corr_svd))
    print('Time Taken = {}'.format(t1-t0))

    t2 = time.time()
    
    
    result_svd_90 = collaborative_basic(trainData, testData, svd_90_matrix_correlation, K)
    RMSE_svd_90, SR_corr_svd_90, precisionTopK_svd_90 = compuation(result_svd_90, testData)
    t3 = time.time()
    print('\n')
    
    print('SVD 90% energy Result')
    print('RMSE = {}'.format(RMSE_svd_90))
    print('Precision on top K = {}'.format(precisionTopK_svd_90))
    print('Spearman Rank Correlation = {}'.format(SR_corr_svd_90))
    print('Time Taken = {}'.format(t3-t2))

    

if __name__ == '__main__':
    main()
