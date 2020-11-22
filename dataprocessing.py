import pandas as pd
import numpy as np


def array_convert(data, minu_index, maxu_index, t_movies):
    combine = []
    for userId in range(minu_index, maxu_index + 1):
        movieID = data[:, 1][data[:, 0] == userId]
        ratingsID = data[:, 2][data[:, 0] == userId]
        ratings = np.zeros(t_movies)
        ratings[movieID - 1] = ratingsID
        combine.append(list(ratings))
    return combine

      
def main():
    training_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    training_set = np.array(training_set, dtype='int')
    test_set = np.array(test_set, dtype='int')
    t_user= int(max(max(training_set[:,0]), max(test_set[:,0])))
    t_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
    
    
    minu_index = 1
    maxu_index = t_user
    training_set = array_convert(training_set, minu_index, maxu_index, t_movies)
    test_set = array_convert(test_set, minu_index, maxu_index, t_movies)
    training_data = np.array([np.array(x) for x in training_set])
    test_data = np.array([np.array(x) for x in test_set])
    np.save('train.npy',training_data)
    np.save('test.npy', test_data)
    
if __name__ == '__main__':
    main()
    
    
    
    