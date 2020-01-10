from KMeans import KMeans
from BisectingKMeans import BisectingKMeans
from readDataBase import ReadDataBase
from sklearn.decomposition import PCA

if __name__ == '__main__':
    # Read the players from the file
    dataBase = ReadDataBase()
    dataBase.chooseTeams(["Ajax", "PSV", "Feyenoord", "AZ Alkmaar", "Vitesse", "FC Utrecht",
    "Heracles Almelo", "FC Groningen", "ADO Den Haag", "Willem II"])

    # Set a preset here
    # Supported presets are: "Regular", "Wings"
    preset = "Regular"
    if preset == "Regular":
        clusters = 4
    elif preset == "Wings":
        clusters = 3
    
    dataBase.run(preset)

    # Make the list of players and certain attributes
    bigAttributes = ["ShortPassing", "LongPassing", "SprintSpeed", "StandingTackle", "SlidingTackle", "Finishing", "Acceleration", "Crossing", "Dribbling", "GKPositioning"]
    wingAttributes = ["Acceleration", "Crossing", "Dribbling", "SprintSpeed", "ShortPassing", "LongPassing", "GKPositioning", "Agility"]


    # Set this to what attribute list you want to use
    if preset == "Regular":
        attributeList = bigAttributes
    elif preset == "Wings":
        attributeList = wingAttributes

    X = dataBase.makePlayersList(attributeList)
    print("Working with", len(X), "players with", len(attributeList), "attributes each.")
    kX = X.copy()
    bkX = X.copy()

    # Clustering with KMeans
    kmeans = KMeans()
    kmeans.setK(clusters)
    kmeans.fit(kX)
    pred = kmeans.predict(kX)

    # Make plot for KMeans

    # Convert data points to 2D points for plotting
    pca = PCA(n_components=2)
    kX = pca.fit_transform(kX)

    # Make labels based on player position
    """
    labels for regular:
        0 - goalkeepers
        1 - defenders
        2 - midfielders
        3 - forwards
        
    labels for wingers:
        0 - goalkeepers
        1 - wingers
        2 - centers
    """
    labels = []
    if preset == "Regular":
        for index in range(len(kX)):
            if index in dataBase.goalkeepers:
                labels.append(0)
            else:
                if index in dataBase.defenders:
                    labels.append(1)
                else:
                    if index in dataBase.midfielders:
                        labels.append(2)
                    else:
                        labels.append(3)

    elif preset == "Wings":
        for index in range(len(kX)):
            if index in dataBase.goalkeepers:
                labels.append(0)
            else:
                if index in dataBase.wings:
                    labels.append(1)
                else:
                    labels.append(2)

    # Plot the results
    kmeans.plot(labels,pred, kX)

    # Clustering with Bisecting KMeans
    bkmeans = BisectingKMeans()
    bkmeans.setK(clusters)
    bkmeans.setNrIterations(10)
    bkmeans.fit(bkX)
    pred = bkmeans.predict(bkX)

    # Make plot for KMeans

    # Convert data points to 2D points for plotting
    pca = PCA(n_components=2)
    bkX = pca.fit_transform(bkX)

    # Make labels based on player position
    """
    labels for regular:
        0 - goalkeepers
        1 - defenders
        2 - midfielders
        3 - forwards
        
    labels for wingers:
        0 - goalkeepers
        1 - wingers
        2 - centers
    """
    labels = []
    if preset == "Regular":
        for index in range(len(bkX)):
            if index in dataBase.goalkeepers:
                labels.append(0)
            else:
                if index in dataBase.defenders:
                    labels.append(1)
                else:
                    if index in dataBase.midfielders:
                        labels.append(2)
                    else:
                        labels.append(3)
    elif preset == "Wings":
        for index in range(len(bkX)):
            if index in dataBase.goalkeepers:
                labels.append(0)
            else:
                if index in dataBase.wings:
                    labels.append(1)
                else:
                    labels.append(2)

    # Plot the results
    kmeans.plot(labels, pred, bkX)

