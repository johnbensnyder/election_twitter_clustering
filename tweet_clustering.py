'''

This code reads in the collection of all twitter data, 
creates clusters based on hashtags, and outputs plots
and term lists of the major clusters for all users, Trump
and Clinton followers, and by quarter of the year.

'''
import os

from ast import literal_eval

import pandas as pd

import numpy as np

from string import punctuation

import re

import matplotlib.pyplot as plt

from sklearn import metrics

def strip_punctuation(string):
    return ''.join(char for char in string if char not in punctuation)
    
#os.chdir('/Users/johnsnyder/Documents/Fall_2016/cs505/final_project')

all_tweets = pd.read_csv('all_tweets_by_day_month.csv')

all_tweets['text'] = [re.findall(r"#(\w+)", str(s)) for s in all_tweets['text']]

all_tweets = all_tweets.loc[[True if len(i)>0 else False for i in all_tweets['text']]]
               
tweets_by_user = all_tweets[['user_ids','text']]

tweets_by_user['text'] = [' '.join(hashtag) for hashtag in tweets_by_user['text']]

tweets_by_user['text'] = [a_tweet + ' ' for a_tweet in tweets_by_user['text']]

#hashtags_by_user['text'] = [' '.join(a_line)+' ' for a_line in hashtags_by_user['text']]

tweets_by_user = tweets_by_user.groupby('user_ids').sum().reset_index()

categories = all_tweets[['user_ids','target']].drop_duplicates('user_ids')

tweets_by_user = tweets_by_user.merge(categories,on='user_ids')

n_features = 2000

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                   
tfidf = tfidf_vectorizer.fit_transform(tweets_by_user['text'])                                   

def evaluate_clusters(X,max_clusters,filename):
    error = np.zeros(max_clusters+1)
    error[0] = 0;
    for k in range(1,max_clusters+1):
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=300)
        kmeans.fit_predict(X)
        error[k] = kmeans.inertia_

    plt.plot(range(1,len(error)),error[1:])
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')
    plt.savefig(filename, dpi = 300)

evaluate_clusters(tfidf,10,'cluster_eval_1.png')



def ri_evaluate_clusters(X,max_clusters,ground_truth,filename):
    ri = np.zeros(max_clusters+1)
    ri[0] = 0;
    for k in range(1,max_clusters+1):
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans.fit_predict(X)
        ri[k] = metrics.adjusted_rand_score(kmeans.labels_,ground_truth)
    plt.plot(range(1,len(ri)),ri[1:])
    plt.xlabel('Number of clusters')
    plt.ylabel('Adjusted Rand Index')
    plt.savefig(filename)
    
ri_evaluate_clusters(tfidf,10,tweets_by_user['target'],'rand_index.png')

'''
so let's try 3, 6, and 9
'''

'''
silhouette_score takes too much memory
def sc_evaluate_clusters(X,max_clusters):
    s = np.zeros(max_clusters+1)
    s[0] = 0;
    s[1] = 0;
    for k in range(2,max_clusters+1):
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans.fit_predict(X)
        s[k] = metrics.silhouette_score(X,kmeans.labels_,metric='cosine')
    plt.plot(range(2,len(s)),s[2:])
    plt.xlabel('Number of clusters')
    plt.ylabel('Adjusted Rand Index')

sc_evaluate_clusters(tfidf,10)
'''
k=3

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')
    
'''
write to file
'''
with open('k3_terms.txt','w') as outfile:
    outfile.write('Top Terms per cluster k=6')
    asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
    order_centroids = asc_order_centroids[:,::-1]
    terms = tfidf_vectorizer.get_feature_names()
    for i in range(k):
        outfile.write("\nCluster {}:".format(i))
        for ind in order_centroids[i, :10]:
            outfile.write('\n {}'.format(terms[ind]))
    print('\n')

np.bincount(clusters)

'''
looks like the political group in general forms a cluster
'''

'''for visualizing, use a smaller subset of the data based on the cluster'''
tweet_data_3_cluster = tweets_by_user.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.01) if i == 2 else np.random.binomial(1,.1) for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]
 
tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 
'''
tfidf_vectorizer_reduced = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                                                    
tfidf_reduced = tfidf_vectorizer_reduced.fit_transform(tweet_data_3_cluster['text'])                                   
'''
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_reduced)


import matplotlib as mpl

from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cols = [['g','r','b'][l] for l in tweet_data_3_cluster['target']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('K = 3 by following')
plt.savefig('clustering_all_k_3_by_follower.png',dpi=150)

cols = [['g','r','b'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('K = 3 by cluster')
plt.savefig('clustering_all_k_3_by_cluster.png')


'''
so this isn't too clear
it looks like we can't really identify based on followers
'''


'''try with 6 clusters'''


k=4

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')
    
'''
write to file
'''
with open('k4_terms.txt','w') as outfile:
    outfile.write('Top Terms per cluster k=6')
    asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
    order_centroids = asc_order_centroids[:,::-1]
    terms = tfidf_vectorizer.get_feature_names()
    for i in range(k):
        outfile.write("\nCluster {}:".format(i))
        for ind in order_centroids[i, :10]:
            outfile.write('\n {}'.format(terms[ind]))
    print('\n')

np.bincount(clusters)

'''
the last cluster looks very pro trump
'''

'''for visualizing, use a smaller subset of the data based on the cluster'''
tweet_data_3_cluster = tweets_by_user.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.005) if i == 1 else np.random.binomial(1,.1) for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]
 
tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 
'''
tfidf_vectorizer_reduced = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                                                    
tfidf_reduced = tfidf_vectorizer_reduced.fit_transform(tweet_data_3_cluster['text'])                                   
'''
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_reduced)


import matplotlib as mpl

from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cols = [['g','r','b'][l] for l in tweet_data_3_cluster['target']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('K = 4 by following')
plt.savefig('clustering_all_k_4_by_follower.png',dpi=150)


cols = [['g','b','k','r'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('K = 4 by cluster')
plt.savefig('clustering_all_k_4_by_cluster.png',dpi=150)


'''try with 8 clusters'''


k=9

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

'''
write to file
'''
with open('k9_terms.txt','w') as outfile:
    outfile.write('Top Terms per cluster k=6')
    asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
    order_centroids = asc_order_centroids[:,::-1]
    terms = tfidf_vectorizer.get_feature_names()
    for i in range(k):
        outfile.write("\nCluster {}:".format(i))
        for ind in order_centroids[i, :10]:
            outfile.write('\n {}'.format(terms[ind]))
    print('\n')


np.bincount(clusters)



'''for visualizing, use a smaller subset of the data based on the cluster'''
tweet_data_3_cluster = tweets_by_user.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.005) if i == 3 else np.random.binomial(1,.1) for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]
 
tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 
'''
tfidf_vectorizer_reduced = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                                                    
tfidf_reduced = tfidf_vectorizer_reduced.fit_transform(tweet_data_3_cluster['text'])                                   
'''
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_reduced)


import matplotlib as mpl

from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cols = [['g','r','b'][l] for l in tweet_data_3_cluster['target']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('K = 9 by following')
plt.savefig('clustering_all_k_9_by_follower.png')


cols = [['k','r','k','g','g','k','b','k','k'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('K = 9 by cluster')
plt.savefig('clustering_all_k_9_by_cluster.png')







'''now try at different times'''
'''first quarter'''

all_tweets = pd.read_csv('all_tweets_by_day_month.csv')

all_tweets['text'] = [re.findall(r"#(\w+)", str(s)) for s in all_tweets['text']]

all_tweets = all_tweets.loc[[True if len(i)>0 else False for i in all_tweets['text']]]

all_tweets_q1 = all_tweets.loc[all_tweets['month']<4]

tweets_by_user_q1 = all_tweets_q1[['user_ids','text']]

tweets_by_user_q1['text'] = [' '.join(hashtag) for hashtag in tweets_by_user_q1['text']]

tweets_by_user_q1['text'] = [a_tweet + ' ' for a_tweet in tweets_by_user_q1['text']]

tweets_by_user_q1 = tweets_by_user_q1.groupby('user_ids').sum().reset_index()

tweets_by_user_q1 = tweets_by_user_q1.merge(categories,on='user_ids')

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                   
tfidf = tfidf_vectorizer.fit_transform(tweets_by_user_q1['text'])                                   

k=3

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

np.bincount(clusters)

tweet_data_3_cluster = tweets_by_user_q1.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.9) if i != 1 else np.random.binomial(1,.01) if i==1 else np.random.binomial(1,.1)  for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]

tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 

dist = 1 - cosine_similarity(tfidf_reduced)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cols = [['g','r','b'][l] for l in tweet_data_3_cluster['target']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('K = 3 by follower')
plt.savefig('clustering_early_k_3_by_follower.png')

cols = [['g','r','b'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)

'''early with 5'''

k=5

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

np.bincount(clusters)

'''not much here, try later'''
'''second quarter'''

all_tweets = pd.read_csv('all_tweets_by_day_month.csv')

all_tweets['text'] = [re.findall(r"#(\w+)", str(s)) for s in all_tweets['text']]

all_tweets = all_tweets.loc[[True if len(i)>0 else False for i in all_tweets['text']]]

all_tweets_q2 = all_tweets.loc[all_tweets['month']<7]

all_tweets_q2 = all_tweets_q2.loc[all_tweets['month']>3]

tweets_by_user_q2 = all_tweets_q2[['user_ids','text']]

tweets_by_user_q2['text'] = [' '.join(hashtag) for hashtag in tweets_by_user_q2['text']]

tweets_by_user_q2['text'] = [a_tweet + ' ' for a_tweet in tweets_by_user_q2['text']]

tweets_by_user_q2 = tweets_by_user_q1.groupby('user_ids').sum().reset_index()

tweets_by_user_q2 = tweets_by_user_q1.merge(categories,on='user_ids')

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                   
tfidf = tfidf_vectorizer.fit_transform(tweets_by_user_q1['text'])                                   

k=3

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

np.bincount(clusters)

tweet_data_3_cluster = tweets_by_user_q2.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.9) if i != 1 else np.random.binomial(1,.01) if i==1 else np.random.binomial(1,.1)  for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]

tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 

dist = 1 - cosine_similarity(tfidf_reduced)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cols = [['g','r','b'][l] for l in tweet_data_3_cluster['target']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('K = 3 by follower')
plt.savefig('clustering_early_k_3_by_follower.png')

cols = [['g','r','b'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)

'''early with 5'''

k=5

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

np.bincount(clusters)

'''third quarter'''


all_tweets = pd.read_csv('all_tweets_by_day_month.csv')

all_tweets['text'] = [re.findall(r"#(\w+)", str(s)) for s in all_tweets['text']]

all_tweets = all_tweets.loc[[True if len(i)>0 else False for i in all_tweets['text']]]

all_tweets_q3 = all_tweets.loc[all_tweets['month']<10]

all_tweets_q3 = all_tweets_q3.loc[all_tweets['month']>6]

tweets_by_user_q3 = all_tweets_q3[['user_ids','text']]

tweets_by_user_q3['text'] = [' '.join(hashtag) for hashtag in tweets_by_user_q3['text']]

tweets_by_user_q3['text'] = [a_tweet + ' ' for a_tweet in tweets_by_user_q3['text']]

tweets_by_user_q3 = tweets_by_user_q3.groupby('user_ids').sum().reset_index()

tweets_by_user_q3 = tweets_by_user_q3.merge(categories,on='user_ids')

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                   
tfidf = tfidf_vectorizer.fit_transform(tweets_by_user_q1['text'])                                   

k=3

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

np.bincount(clusters)

tweet_data_3_cluster = tweets_by_user_q3.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.9) if i != 1 else np.random.binomial(1,.01) if i==1 else np.random.binomial(1,.1)  for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]

tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 

dist = 1 - cosine_similarity(tfidf_reduced)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cols = [['g','r','b'][l] for l in tweet_data_3_cluster['target']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('K = 3 by follower')
plt.savefig('clustering_early_k_3_by_follower.png')

cols = [['g','r','b'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)

'''early with 5'''

k=5

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

np.bincount(clusters)





'''forth quarter'''
all_tweets = pd.read_csv('all_tweets_by_day_month.csv')

all_tweets['text'] = [re.findall(r"#(\w+)", str(s)) for s in all_tweets['text']]

all_tweets = all_tweets.loc[[True if len(i)>0 else False for i in all_tweets['text']]]

all_tweets_late = all_tweets.loc[all_tweets['month']>=10]

tweets_by_user_late = all_tweets_late[['user_ids','text']]

tweets_by_user_late['text'] = [' '.join(hashtag) for hashtag in tweets_by_user_late['text']]

tweets_by_user_late['text'] = [a_tweet + ' ' for a_tweet in tweets_by_user_late['text']]

tweets_by_user_late = tweets_by_user_late.groupby('user_ids').sum().reset_index()

tweets_by_user_late = tweets_by_user_late.merge(categories,on='user_ids')

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                   
tfidf = tfidf_vectorizer.fit_transform(tweets_by_user_late['text'])                                   

k=3

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

'''
write to file
'''
with open('k9_terms.txt','w') as outfile:
    outfile.write('Top Terms per cluster k=6')
    asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
    order_centroids = asc_order_centroids[:,::-1]
    terms = tfidf_vectorizer.get_feature_names()
    for i in range(k):
        outfile.write("\nCluster {}:".format(i))
        for ind in order_centroids[i, :10]:
            outfile.write('\n {}'.format(terms[ind]))
    print('\n')

np.bincount(clusters)

tweet_data_3_cluster = tweets_by_user_late.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.002) if i == 0 else np.random.binomial(1,.05) if i==1 else np.random.binomial(1,.1)  for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]

tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 

dist = 1 - cosine_similarity(tfidf_reduced)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cols = [['g','r','b'][l] for l in tweet_data_3_cluster['target']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)

cols = [['g','r','b'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)








k=5

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

np.bincount(clusters)

tweet_data_3_cluster = tweets_by_user_late.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.002) if i == 0 else np.random.binomial(1,.05) if i==1 else np.random.binomial(1,.1)  for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]

tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 

dist = 1 - cosine_similarity(tfidf_reduced)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cols = [['g','r','b'][l] for l in tweet_data_3_cluster['target']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)

cols = [['b','r','g','y','b'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)




'''what about within the candidate followers'''
'''trump first'''

all_tweets = pd.read_csv('all_tweets_by_day_month.csv')

all_tweets['text'] = [re.findall(r"#(\w+)", str(s)) for s in all_tweets['text']]

all_tweets = all_tweets.loc[[True if len(i)>0 else False for i in all_tweets['text']]]

all_tweets_trump = all_tweets.loc[all_tweets['target']==1]

tweets_by_user_trump = all_tweets_trump[['user_ids','text']]

tweets_by_user_trump['text'] = [' '.join(hashtag) for hashtag in tweets_by_user_trump['text']]

tweets_by_user_trump['text'] = [a_tweet + ' ' for a_tweet in tweets_by_user_trump['text']]

tweets_by_user_trump = tweets_by_user_trump.groupby('user_ids').sum().reset_index()

tweets_by_user_trump = tweets_by_user_trump.merge(categories,on='user_ids')

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                   
tfidf = tfidf_vectorizer.fit_transform(tweets_by_user_trump['text'])                                   

evaluate_clusters(tfidf,10,'eval_cluster_trump.png')
ri_evaluate_clusters(tfidf,10,tweets_by_user_trump['target'],'rand_index_trump.png')

k=5

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

'''
write to file
'''
with open('k5_terms_trump.txt','w') as outfile:
    outfile.write('Top Terms per cluster k=6')
    asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
    order_centroids = asc_order_centroids[:,::-1]
    terms = tfidf_vectorizer.get_feature_names()
    for i in range(k):
        outfile.write("\nCluster {}:".format(i))
        for ind in order_centroids[i, :10]:
            outfile.write('\n {}'.format(terms[ind]))
    print('\n')

np.bincount(clusters)

tweet_data_3_cluster = tweets_by_user_trump.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.01) if i == 4 else np.random.binomial(1,.1) for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]

tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 

dist = 1 - cosine_similarity(tfidf_reduced)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cols = [['k','r','g','b','y'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('Trump Followers K=5')
plt.savefig('Trump_followers.png', dpi = 150)

'''really ardent trump backers only form a small part of his followers'''
'''now with clinton followers'''
all_tweets = pd.read_csv('all_tweets_by_day_month.csv')

all_tweets['text'] = [re.findall(r"#(\w+)", str(s)) for s in all_tweets['text']]

all_tweets = all_tweets.loc[[True if len(i)>0 else False for i in all_tweets['text']]]

all_tweets_clinton = all_tweets.loc[all_tweets['target']==2]

tweets_by_user_clinton = all_tweets_clinton[['user_ids','text']]

tweets_by_user_clinton['text'] = [' '.join(hashtag) for hashtag in tweets_by_user_clinton['text']]

tweets_by_user_clinton['text'] = [a_tweet + ' ' for a_tweet in tweets_by_user_clinton['text']]

tweets_by_user_clinton = tweets_by_user_clinton.groupby('user_ids').sum().reset_index()

tweets_by_user_clinton = tweets_by_user_clinton.merge(categories,on='user_ids')

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=10,
                                   max_features=n_features,
                                   stop_words='english')
                                   
tfidf = tfidf_vectorizer.fit_transform(tweets_by_user_clinton['text'])                                   

evaluate_clusters(tfidf,10,'eval_cluster_clinton.png')
ri_evaluate_clusters(tfidf,10,tweets_by_user_clinton['target'],'rand_index_clinton.png')

k=5

kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
clusters = np.array(kmeans.fit_predict(tfidf))


print("Top terms per cluster:")
asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
order_centroids = asc_order_centroids[:,::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    print('')

'''
write to file
'''
with open('k5_terms_clinton.txt','w') as outfile:
    outfile.write('Top Terms per cluster k=6')
    asc_order_centroids = kmeans.cluster_centers_.argsort()#[:, ::-1]
    order_centroids = asc_order_centroids[:,::-1]
    terms = tfidf_vectorizer.get_feature_names()
    for i in range(k):
        outfile.write("\nCluster {}:".format(i))
        for ind in order_centroids[i, :10]:
            outfile.write('\n {}'.format(terms[ind]))
    print('\n')

np.bincount(clusters)

tweet_data_3_cluster = tweets_by_user_clinton.copy()
tweet_data_3_cluster['clusters'] = clusters
tweet_data_3_cluster['tokeep'] = [np.random.binomial(1,.01) if i == 1 else np.random.binomial(1,.1) for i in clusters]
tweet_data_3_cluster = tweet_data_3_cluster.loc[tweet_data_3_cluster['tokeep']==1]

tfidf_reduced = tfidf[tweet_data_3_cluster.index,:] 

dist = 1 - cosine_similarity(tfidf_reduced)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]

cols = [['k','r','g','b','y'][l] for l in tweet_data_3_cluster['clusters']]
plt.scatter(pos[:, 0], pos[:, 1], s=12, c=cols)
plt.title('Clinton Followers K=5')
plt.savefig('Clinton_followers.png',dpi=150)


'''now get distributions of top terms'''

all_tweets = pd.read_csv('all_tweets_by_day_month.csv')

all_tweets['text'] = [re.findall(r"#(\w+)", str(s)) for s in all_tweets['text']]

all_tweets = all_tweets.loc[[True if len(i)>0 else False for i in all_tweets['text']]]

word = 'spiritcooking'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png',dpi=200)
plt.show()


word = 'maga'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png',dpi = 200)
plt.show()



word = 'imwithher'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png',dpi=200)
plt.show()





word = 'trump'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png')
plt.show()




word = 'lockherup'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png')
plt.show()






word = 'nodapl'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png')
plt.show()




word = 'blacklivesmatter'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png')
plt.show()





word = 'trump2016'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png')
plt.show()





word = 'draintheswamp'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png', dpi = 200)
plt.show()


word = 'crookedhillary'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png')
plt.show()




word = 'brexit'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png')
plt.show()






word = 'election2016'

all_tweets['has_word'] = [1 if word in i else 0 for i in all_tweets['text']]

word_dist = all_tweets[['day','has_word']].groupby('day').sum().reset_index()

plt.plot(word_dist['day'],word_dist['has_word'])
plt.xlabel('Day of Year')
plt.ylabel('Daily Count')
plt.title(word + ' frequency')
plt.grid(True)
plt.savefig(word+'.png',dpi=200)
plt.show()