"""
@author Roberta Buzatu s1020137
@author Bart van der Heijden s1017343

"""

import numpy as np 
from random import randrange
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class KMeans:
    """
    Class representation for the KMeans algorithm.
    
    How to use:
        1. set the number of desired clusters with setK(number).
        2. fit the model on the training set with fit(Xtrain).
        3. predict classes for the test set with predict(Xtest).
        4. plot the output with plot(predictedClasses, Xtest, titleForThePlot)
        
    """
    k = 0  # number of clusters
    centroids = []  # list of centroids

    """
    Set the number of desired clusters.
    """
    def setK(self, number):
        self.k = number

    """
    Fit the model on the training data.
    Input:
        Xtrain - list of records and their attributes
    """
    def fit(self, Xtrain):
        # Select random k points as the initial centroids
        centroids = []
        for i in range(self.k):
            index = randrange(len(Xtrain))
            centroids.append(Xtrain[index])
        # Repeat while centroids don't change
        change = 10
        while change > 0:
            change = 0
            # Initialize clusters
            clusters = []
            for i in range(self.k):
                clusters.append([])
            for element in Xtrain:
                # Find the closest centroid
                distances = []
                for centroid in centroids:
                    distances.append(np.linalg.norm(np.array(element)-np.array(centroid)))
                position = distances.index(min(distances))
                # Add the element to the good cluster
                clusters[position].append(element)
            # Compute the new centroids 
            newCentroids = []
            for number in range(len(centroids)):
                newCentroid = np.mean(clusters[number], axis = 0)
                newCentroids.append(newCentroid)
                # Compute the change
                change += np.linalg.norm(np.array(centroids[number]) - np.array(newCentroid))
            # Update the centroids
            centroids.clear()
            centroids = newCentroids.copy()
        # Return the fitted centroids
        self.centroids = centroids.copy()

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
                distances.append(np.linalg.norm(np.array(Xtest[index])-np.array(centroid)))
            position = distances.index(min(distances))
            predictedClass.append(position)
        return predictedClass
    
    """
    Plot the clusters in 3d space.
    Input:
        predictedClass - list of predicted classes 
        Xtest - list of records and their attributes
        title - title for the plot
    """
    def plot3d(self, predictedClass, Xtest, title):
        x = []
        y = []
        z = []
        for element in Xtest:
            x.append(element[0])
            y.append(element[1])
            z.append(element[2])
               
        fig, ax = plt.subplots()
        ax = plt.axes(projection = '3d')
        scatter = ax.scatter(x, y, z, c=predictedClass)
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        
        plt.xlabel('Attribute A')
        plt.ylabel('Attribute B')
        plt.title(title)
        plt.show()

    """
    Plot the clusters.
    Input:
        labels - list of labels based on player position
        predictedClass - list of predicted classes 
        Xtest - list of records and their attributes
    """

    def plot(self, labels, predictedClass, Xtest):
        x = []
        y = []
        for element in Xtest:
            x.append(element[0])
            y.append(element[1])

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        ax = fig.add_subplot(1,2,1)
        scatter = ax.scatter(x, y, c=predictedClass)
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        plt.title("Player Clustering")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        ax = fig.add_subplot(1, 2, 2)
        scatter = ax.scatter(x, y, c=labels)
        legend = ax.legend(*scatter.legend_elements(), title="Labels")
        ax.add_artist(legend)
        plt.title("Player positions")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()