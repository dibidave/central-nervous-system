import numpy as np

def get_data():
    data = np.loadtxt("data/data.txt", delimiter='\t', dtype=np.int16)
    return data

def get_training_data():
    data = np.loadtxt("data/train.txt", delimiter='\t', dtype=np.int16)
    return data

def get_test_data():
    data = np.loadtxt("data/test.txt", delimiter='\t', dtype=np.int16)
    return data

def get_movies():
    movie_id = np.loadtxt("data/movies.txt", delimiter='\t', usecols=0, 
                          encoding='iso 8859-1', dtype=np.int16)
    movie_title = np.loadtxt("data/movies.txt", delimiter='\t', usecols=1, 
                          encoding='iso 8859-1', dtype=object)
    movie_genre = np.loadtxt("data/movies.txt", delimiter='\t', usecols=np.arange(2, 21), 
                          encoding='iso 8859-1', dtype=np.int16)
    genres = np.array(['Unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 
                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    return movie_id, movie_title, movie_genre, genres

def list_to_matrix(Y, M=943, N=1682, fill=np.nan):
    '''
    Convert the list of rating with shape (?, 3) into a matrix of rating with shape (M, N)
    and empty values filled with NaN by default
    '''
    
    YM = np.full((M, N), fill)
    for l in Y:
        YM[l[0]-1, l[1]-1] = l[2]
    return YM