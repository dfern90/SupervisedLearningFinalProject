import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from ast import literal_eval
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

#This first block imports our data
#Adjust the paths as needed
df1 = pd.read_csv('data/tmdb_5000_movies.csv')
df2 = pd.read_csv('data/tmdb_5000_credits.csv')
extract_cast = df2["cast"]
extract_crew = df2["crew"]
df1 = df1.join(extract_cast)
df = df1.join(extract_crew)
df = df[['title', 'keywords', 'genres', 'cast', 'crew', 'overview', 'popularity', 'vote_average', 'vote_count','id']]
df = df.dropna()
features = ["keywords", "genres", "cast","crew"]

#top_five extracts the top 5 names for each entry in our string list
def top_five(x):
    if isinstance(x, list):
        items = [i["name"] for i in x]
        if len(items) > 5:
            items = items[:]
        return items
    return []

#Traverses the list of strings in our columns
#and gets the top 5 in each entry
for feature in features:
    df[feature] = df[feature].apply(literal_eval)
for feature in features:
    df[feature] = df[feature].apply(top_five)
    df[feature] = [','.join(map(str, l)) for l in df[feature]]

#Makes a soup, which is all the metadata from our selected features
countv_df = df.copy()
def make_soup(features):
    return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['crew'] + ' ' + ' '.join(features['genres'])
countv_df["soup"] = countv_df.apply(make_soup, axis=1)
print(countv_df["soup"].head())

#Initializes count_vectorizer
#stop_words are words like 'a' and 'the'
count_vectorizer = CountVectorizer(stop_words="english")
matrix = count_vectorizer.fit_transform(countv_df["soup"])
cosine = cosine_similarity(matrix, matrix)
countv_df = countv_df.reset_index()
indices = pd.Series(countv_df.index, index=countv_df['title'])

#Our first model, gets recommendations by entering a movietitle
# and the cosine_similarity.
#Returns ten movie recommendations and the similarity score
def get_recommendations(title, cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    movies_indices = [ind[0] for ind in sim_scores]
    movies = countv_df["title"].iloc[movies_indices]
    s_score = pd.DataFrame(sim_scores, columns=['id', 'score'])

    return movies, s_score['score']

print(get_recommendations("Minions", cosine))

#Our second model, gets recommendations by entering a movietitle
#the cosine_similarity, and the quantile for vote_count
#Returns ten movie recommendations sorted by vote_average
def enhanced_recommendations(title, cosine_sim, q):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]

    movie_indices = [i[0] for i in sim_scores]
    movies = countv_df.iloc[movie_indices][['title', 'vote_count', 'vote_average']]

    count = movies[movies['vote_count'].notnull()]['vote_count']

    frame = movies[(movies['vote_count'] >= count.quantile(q)) & (movies['vote_count'].notnull()) & (
        movies['vote_average'].notnull())]
    frame = frame.sort_values('vote_average', ascending=False).head(10)

    return frame

print(enhanced_recommendations("Minions", cosine, .6))