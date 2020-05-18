import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def build_model(pca_n, n_clusters):
    '''Creates a pipeline for doing KMeans clustering

    Args:
        pca_n (int): number of pca components
        n_clusters (int): number of clusters

    Returns:
        pipeline (sklearn.pipeline.Pipeline)
    '''
    pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(pca_n)),
            ('kmeans', KMeans(n_clusters=n_clusters))
    ])

    return pipeline

if __name__ == '__main__':
    '''Fits a clustering model and saves pipeline to a pickle file
    with name 'clust_model' + str(n_clusters) + '.pkl'

    Args:
        cleandata_filepath (str): filepath of cleaned data
        pca_n (int): number of pca components
        n_clusters (int): number of clusters
    '''

    pca_n = int(60)
    n_clusters = int(8)

    print('Loading data...')
    clean_df = pd.read_csv('azdias_clean.csv')
    clean_df.drop('Unnamed: 0', axis=1, inplace=True)

    print('Building model...')
    model = build_model(pca_n, n_clusters)

    print('Fitting model...')
    model.fit(clean_df)

    print('Saving model...')
    f = open('clust_model' + str(n_clusters) + '.pkl', 'wb')
    pickle.dump(model, f)    
