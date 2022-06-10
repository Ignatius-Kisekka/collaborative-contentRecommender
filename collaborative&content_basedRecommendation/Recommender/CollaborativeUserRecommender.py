import pandas as pd

# To download the data:

# Download the small version of the ml-latest.zip file from MovieLens Latest DataSets. ie https://grouplens.org/datasets/movielens/latest/
# Unzip the file. The files you'll be using are movies.csv and ratings.csv.

ratings = pd.read_csv(r'..\ratings.csv')
movies = pd.read_csv(r'..\movies.csv')

# Creating a user dataframe from user_input_list
# 1
def user_dataframe(user_list):
    user_df = pd.DataFrame(user_list)
    user_df = user_df.merge(movies)
    user_df.drop('title', axis=1, inplace=True)
    user_df = user_df.sort_values('movieId')
    user_df = user_df.reset_index(drop=True)

    # Removing common movie titles
    # for title_1 in user_df['title']:
    #     for title_2 in user_df['title']:
    #         if title_1 == title_2:
    #             index_list = user_df.index[user_df['title'] == title_2].tolist()
    #             if len(index_list) > 1:
    #                 user_df.drop(user_df.index[user_df['movieId'] == user_df.iloc[index_list[1]]['movieId']], axis=0,
    #                              inplace=True)
    #                 user_df.reset_index(inplace=True)

    return user_df


# 2
# Getting similar users
def df_users_with_sim_movies(user_df):
    # Creating dataframe
    df = ratings[ratings['movieId'].isin(user_df['movieId'])]

    # Grouping df by userId
    sim_user_groups = df.groupby('userId')

    # sorting groups in ascending order ie most movies in common
    sim_user_groups = sorted(sim_user_groups, key=lambda x: len(x[1]), reverse=True)

    similar_100_users = sim_user_groups[:100]

    return similar_100_users


# 3
# Computing the level of similarity of the similar users with target user
# Using pearson correlation coefficient
def pearsonC(user_input, sim_user_groups):
    from math import sqrt

    pearsonDict = {}
    user_input.sort_values('movieId', inplace=True)
    for name, group in sim_user_groups:
        # ut == 'target user' ,  ug == 'user in group'
        group.sort_values('movieId', inplace=True)
        n_movies = float(len(group))

        # Creating dataframe from user_dataframe containing common movies between ut && ug
        temp_user_df = user_input[user_input['movieId'].isin(group['movieId'].tolist())]

        # Creating rating column into list
        temp_user_lst = temp_user_df['rating'].tolist()
        temp_group_lst = group['rating'].tolist()

        S_ut = sum(pow(i, 2) for i in temp_user_lst) - pow(sum(i for i in temp_user_lst), 2) / n_movies
        S_ug = sum(pow(i, 2) for i in temp_group_lst) - pow(sum(i for i in temp_group_lst), 2) / n_movies

        S_g_t = sum(i * j for i, j in zip(temp_user_lst, temp_group_lst)) - sum(i for i in temp_user_lst) * sum(
            j for j in temp_group_lst) / n_movies

        if S_ut != 0 and S_ug != 0:
            pearsonDict[name] = S_g_t / sqrt(S_ut * S_ug)

        else:
            pearsonDict[name] = 0

    # Converting pearsonDict into datafame
    pearsonDF = pd.DataFrame.from_dict(pearsonDict, orient='index')
    pearsonDF.columns = ['similarity_index']
    pearsonDF = pearsonDF[pearsonDF['similarity_index'] > 0]

    pearsonDF['userId'] = pearsonDF.index
    pearsonDF = pearsonDF.sort_values('similarity_index', ascending=False)
    pearsonDF.index = range(len(pearsonDF))

    # Return the top 50 with biggest similarity index
    return pearsonDF[:50]


# Getting movies seen by similar users for recommendation
# to avoid the recommending movie that might be seen by one user who rated it highly & rated very low by other users
# we give movies an average weighted score


# 4
def recommender_dataframe(similarityDF):
    movies_to_recommend = pd.merge(similarityDF, ratings)
    movies_to_recommend['weighted_rating'] = movies_to_recommend['similarity_index'] * movies_to_recommend['rating']

    recommendDF = movies_to_recommend.groupby('movieId').sum()[['similarity_index', 'weighted_rating']]
    recommendDF.columns = ['sum_similarity_index', 'sum_weighted_rating']
    recommendDF['movieId'] = recommendDF.index

    # recommendDF['average_weighted_rating'] = recommendDF['sum_weighted_rating'] / recommendDF['sum_similarity_index']

    recommendDF.sort_values(by='sum_weighted_rating', ascending=False, inplace=True)
    recommendDF.index = range(len(recommendDF))
    recommendDF = recommendDF[['movieId', 'sum_similarity_index', 'sum_weighted_rating']]
    return recommendDF


# 5
def movies_recommended(movie_recommending_dataframe):
    movie_R_df = pd.merge(movie_recommending_dataframe, movies)
    movie_R_df = movie_R_df[['movieId']]

    return movie_R_df.head(100)

