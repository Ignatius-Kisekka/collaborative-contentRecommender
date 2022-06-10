
import CollaborativeUserRecommender
import MovieTypeRecommender



# Test user input
user_input_list = [
    {'title': 'Shaggy Dog, The', 'rating': 4.0},
    {'title': 'That Darn Cat!', 'rating': 4.5},
    {'title': 'Heidi Fleiss: Hollywood Madam', 'rating': 3.0},
    {'title': 'Grumpier Old Men', 'rating': 5.0},
    {'title': 'Waiting to Exhale', 'rating': 5.0}
]
# Creating dataframe
dataframe = CollaborativeUserRecommender.user_dataframe(user_input_list)

# Getting similar 100 users sharing preferences
similar_users = CollaborativeUserRecommender.df_users_with_sim_movies(dataframe)

# Calculating their level of similarity using Pearson Correlation Coefficient & taking the top 50
pearsonDF = CollaborativeUserRecommender.pearsonC(dataframe, similar_users)


# Creating dataframe of recommendations basing on the calculated similarity
# CF for 'Collaborative Filtering'

CF_recommendationDF = CollaborativeUserRecommender.recommender_dataframe(pearsonDF)


# Getting the top 100 recommended movies in dataframe
CF_100_movies = CollaborativeUserRecommender.movies_recommended(CF_recommendationDF)
print(CF_100_movies)

# The 100 movies recommended by collaborative filtering are then compared with the movies watched by the target user
# to get the better recommendations

# Creating a user profile based on the genres
uzer_profile = MovieTypeRecommender.user_profile(dataframe)

# Identifying the movies recommended
# IT for 'Item-Item Filtering'

IT_recommended_movies = MovieTypeRecommender.movie_recommend(uzer_profile, CF_100_movies)

# titles of recommended movies
movie_titlez = MovieTypeRecommender.titles_recommended(IT_recommended_movies)

# print(movie_titlez)