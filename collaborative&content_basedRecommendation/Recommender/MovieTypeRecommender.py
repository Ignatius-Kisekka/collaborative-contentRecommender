import pandas as pd
from CollaborativeUserRecommender import movies

# Giving movies recommended their genres
movies_genres = pd.read_csv(r'D:\Data\edx\moviedataset\ml-latest\movies_genres.csv')


# Returns  quantification of the opinion of the user towards every movie genre
def user_profile(input_df):
    user_rating = input_df['rating']

    # Giving user's movies their genres
    user_movie_genre = movies_genres[movies_genres['movieId'].isin(input_df['movieId'].tolist())]
    user_movie_genre = user_movie_genre.drop('movieId', axis=1)
    user_movie_genre.index = range(len(user_movie_genre))

    userz_profile = user_movie_genre.transpose().dot(user_rating)
    return userz_profile


# Identify the movies to be recommended from the top 100 recommended by collaborative filtering
def movie_recommend(userProfile, CF_top_100):
    CF_top_100 = movies_genres[movies_genres['movieId'].isin(CF_top_100['movieId'].to_list())]
    CF_top_100 = CF_top_100.set_index('movieId', drop=True)

    recommendationDF = (CF_top_100 * userProfile).sum(axis=1) / userProfile.sum()
    recommendationDF = recommendationDF.to_frame()
    recommendationDF.columns = ['Weighted score']
    recommendationDF['movieId'] = recommendationDF.index
    recommendationDF = recommendationDF.sort_values('Weighted score', ascending=False)
    recommendationDF = recommendationDF[['movieId', 'Weighted score']]
    recommendationDF.index = range(len(recommendationDF))

    return recommendationDF.head(25)


# Get the titles of recommended movies
def titles_recommended(moviesRecommended):
    moviesRecommended = movies[movies['movieId'].isin(moviesRecommended['movieId'].tolist())]
    return moviesRecommended
