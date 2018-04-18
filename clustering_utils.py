def clustering_accuracy(y_true, y_pred):

    import numpy as np
    """
    Calculate clustering accuracy. Require scikit-learn installed

    See: Unsupervised Deep Embedding for Clustering Analysis

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clustering_evaluation(x, y_true, y_pred):
    import numpy as np
    from sklearn import metrics

    acc = clustering_accuracy(y_true, y_pred)
    homogeneity = metrics.homogeneity_score(y_true, y_pred)

    # Silhouette
    if len(np.unique(y_pred)) > 1:
        sil_cosine = metrics.silhouette_score(x, y_pred, metric='cosine')
        sil_euclidean = metrics.silhouette_score(x, y_pred, metric='euclidean')
    else:
        sil_cosine = -1
        sil_euclidean = 0

    return acc, homogeneity, sil_cosine, sil_euclidean


def clustering(x, y):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    y_pred, db_n_clusters, components = dbscan(x, y)
    db_acc, db_homo, db_sil_cosine, db_sil_euclidean = clustering_evaluation(x, y, y_pred)

    k_n_clusters = len(np.unique(y))
    y_pred, _ = kmeans(x, k_n_clusters)
    k_acc, k_homo, k_sil_cosine, k_sil_euclidean = clustering_evaluation(x, y, y_pred)

    df = pd.DataFrame([['DBScan', 'Accuracy', db_acc], ['DBScan', 'Homogeneity', db_homo],
                       ['DBScan', 'Silhouette Cosine', db_sil_cosine],
                       ['DBScan', 'Silhouette Euclidean', db_sil_euclidean],
                       ['KMeans', 'Accuracy', k_acc], ['KMeans', 'Homogeneity', k_homo],
                       ['KMeans', 'Silhouette Cosine', k_sil_cosine],
                       ['KMeans', 'Silhouette Euclidean', k_sil_euclidean]],
                      columns=['Algorithm', 'Metric', 'Value'])

    ax = df.pivot('Algorithm', 'Metric', 'Value').plot(kind='bar', ax=plt.gca(), ylim=[-1, 1])
    ax.set_xticklabels(["DBScan", "KMeans"], rotation=0)


def dbscan(x, y):
    """
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN

        Returns:
            - Predicted labels
            - Number of clusters in labels, ignoring noise (outliers) if present.
            - Copy of each core sample found by training.
    """

    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.3, min_samples=10).fit(x)

    y_pred = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(y)) - (1 if -1 in y else 0)

    return y_pred, n_clusters, db.components_


def kmeans(x, n_clusters):
    """
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

        Returns:
            - Predicted labels
            - Coordinates of cluster centers
    """

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)

    return kmeans.labels_, kmeans.cluster_centers_


def eval_clustering(x, y):

    import numpy as np

    #y_pred, db_n_clusters, components = dbscan(x, y)
    #db_acc, db_homo, db_sil_cosine, db_sil_euclidean = clustering_evaluation(x, y, y_pred)

    k_n_clusters = len(np.unique(y))
    y_pred, _ = kmeans(x, k_n_clusters)
    k_acc, k_homo, k_sil_cosine, k_sil_euclidean = clustering_evaluation(x, y, y_pred)

    return k_acc, k_homo, k_n_clusters
