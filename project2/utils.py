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

def genre_similarity(genres, fill=np.nan):
    '''
    Given a list of movies with shape (#movies, #genres), caculate the probability of any movie
    belonging to the two genres simultaeneously.
    
    returns a upper triagular matrix with shape (#genres, #genres)
    '''
    M = genres.shape[0]
    N = genres.shape[1]
    S = np.full((N, N), fill)
    for i in range(N):
        for j in range(i, N):
            S[i, j] = np.sum(genres[:, i]*genres[:, j])/M
    return S

def bayesian_rating(Y, thr=5, M=943, N=1682):
    '''
    Correct the rating of the movies by the number of the ratings through a bayesian approach.
    Reference: 
    https://stats.stackexchange.com/questions/15979/how-to-find-confidence-intervals-for-ratings
    
    returns the number of ratings, the original avarage rating and the correctd avarage rating.
    '''
    YM = list_to_matrix(Y, M, N)
    C = np.mean(Y[:, 2]) # Mean rating of all movies
    counts = np.sum(1-np.isnan(YM), axis=0) # Number of ratings for each movie
    counts_sum = np.sum(counts)
    ratings = np.nanmean(YM, axis=0) # Mean rating of the movies
    ratings_all = np.full((N,), C)
    weights = counts/(counts+thr)
    ratings_bayesian = weights*ratings+(1-weights)*ratings_all
    return counts, ratings, ratings_bayesian