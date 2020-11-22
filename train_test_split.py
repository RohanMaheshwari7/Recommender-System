import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ratings = pd.read_csv('ratings.dat', sep = '::', header = None,
                      engine = 'python', encoding = 'latin-1')
    training_data, test_data = train_test_split(ratings, test_size = 0.2, random_state = 0)
    training_data.to_csv('train.csv', header = None, index = False)
    test_data.to_csv('test.csv', header = None, index = False)
    
if __name__ == '__main__':
    main()