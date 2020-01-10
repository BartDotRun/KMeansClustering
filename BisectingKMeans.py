"""
@author Roberta Buzatu s1020137
@author Bart van der Heijden s1017343

"""

from KMeans import KMeans
import numpy as np
import matplotlib.pyplot as plt


class BisectingKMeans:
    """
    Class representation for the Bisecting KMeans algorithm.

    How to use:
        1. set the number of desired clusters with setK(number).
        2. fit the model on the training set with fit(Xtrain).
        3. predict classes for the test set with predict(Xtest).
        4. plot the output with plot(predictedClasses, Xtest, titleForThePlot)

    """

    k = 0  # number of clusters
    nrIterations = 0  # number of iterations to be used in fitting the model
    centroids = []  # list of centroids

    """
    Set the number of desired clusters.
    """
    def setK(self, number):
        self.k = number

    """
    Set the number of desired iterations.
    """
    def setNrIterations(self, number):
        self.nrIterations = number

    """
    Fit the model on the training data.
    Input:
        Xtrain - list of records and their attributes
    """
    def fit(self, Xtrain):
        clusters = [Xtrain]
        # Split clusters until the number of desired clusters is achieved
        while len(clusters)<self.k:
            # Choose the largest cluster to split
            lengths = [len(cluster) for cluster in clusters]
            split = lengths.index(max(lengths))
            # Find some 2 sub-clusters using KMeans
            cluster = clusters[split]
            possibilities = []
            for iter in range(self.nrIterations):
                possibilities.append(self.findClustering(cluster))
            # Choose clustering with smallest SSE
            SSE = [sum for sum, array in possibilities]

            newClasses = possibilities[SSE.index(min(SSE))][1]
            # Update the clusters
            newClusters = []
            # Add all clusters that have not changed
            for index in range(len(clusters)):
                if not index == split:
                    newClusters.append(clusters[index])
            # Add the two new clusters
            cluster1 = [cluster[index] for index in range(len(cluster)) if newClasses[index] == 0]
            cluster2 = [cluster[index] for index in range(len(cluster)) if newClasses[index] == 1]
            newClusters.append(cluster1)
            newClusters.append(cluster2)
            # Update the clusters
            clusters.clear()
            clusters = newClusters.copy()
        # Update the centroids
        self.centroids = [np.mean(cluster, axis = 0) for cluster in clusters]

    """
    Split a cluster into 2 and compute the Sum of Squared Error.
    Input:
        cluster - cluster to be split up
    Output:
        (sse, predictedClasses) - tuple of the Sum of Squared Error and the classes of the cluster
    """
    def findClustering(self, cluster):
        # Use KMeans to form 2 clusters
        kmeans = KMeans()
        kmeans.setK(2)
        kmeans.fit(cluster)
        predictedClass = kmeans.predict(cluster)
        centroids = kmeans.centroids
        # Compute SSE for the clustering
        sum = 0
        for clusterIndex in range(len(centroids)):
            for element in cluster:
                sum += (np.linalg.norm(np.array(element)-np.array(centroids[clusterIndex])))**2
        return sum, predictedClass

    """
    Predict the clusters of unseen data.
    Input:
        Xtest - list of records and their attributes
    Output:
        predictedClass - list of predicted classes for the elements in Xtest
    """
    def predict(self, Xtest):
        predictedClass = []
        for index in range(len(Xtest)):
            # Find the closest centroid and assign its class to the element
            distances = []
            for centroid in self.centroids:
                distances.append(np.linalg.norm(Xtest[index]-centroid))
            position = distances.index(min(distances))
            predictedClass.append(position)
        return predictedClass


