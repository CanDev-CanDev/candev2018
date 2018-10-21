# based on http://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py

import pandas as pd
import numpy as np
import logging
import sys
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
from sklearn import metrics

# for preprocessing
#from shorttext.utils import standard_text_preprocessor_1


####
# Pre process data by lowercasing, removing punctuation, stemming
####
#docs = np.array(df['answer'])

#preprocessor1 = standard_text_preprocessor_1()
#for i, doc in enumerate(docs):
#    if i % 100 == 0:
#        print("Preprocessing row {}".format(i))
#    docs[i] = preprocessor1(doc)

#df2 = pd.DataFrame(docs)
#df2.columns = ['pre-processed answer']
#pd.DataFrame.to_csv(df2, 'preprocessed.csv', index=False)

use_preprocessed = True

if use_preprocessed:
    df = pd.read_csv("preprocessed.csv")
    data = np.array(df['pre-processed answer'])
else:
    df = pd.read_csv("PSC_Training_Dataset.csv")
    data = np.array(df['answer'])

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=True)
X = vectorizer.fit_transform(data)

# target dimension for LSA
n_components=200

print("Performing dimensionality reduction using LSA")
t0 = time()
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

print("done in %fs" % (time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))

print()

clusters = 5

# Clustering
km = MiniBatchKMeans(n_clusters=clusters, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=True)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()


print("Top terms per cluster:")

original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(clusters):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
