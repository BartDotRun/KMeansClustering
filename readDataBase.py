import pandas
import numpy as np
from scipy.stats import stats


class ReadDataBase:

    data = pandas.read_csv("data.csv")
    teams = []
    indexes = []
    goalkeepers = []
    defenders = []
    midfielders = []
    forwards = []
    positions = []
    wings = []
    center = []


    def run(self, visualiser):
        self.selectPlayersFromTeams()
        if visualiser == "Regular":
            self.findLinePositions()
        elif visualiser == "Wings":
            self.findWingPositions()

    def chooseTeams(self, list):
        self.teams = list

    def selectPlayersFromTeams(self):
        clubs = list(self.data["Club"])
        self.indexes = [index for index in range(len(clubs)) if clubs[index] in self.teams]

    def findLinePositions(self):
        allPositions = list(self.data["Position"])
        clubs = list(self.data["Club"])
        positions = [allPositions[index] for index in range(len(clubs)) if clubs[index] in self.teams]
        self.positions = positions
        self.goalkeepers = [index for index in range(len(positions)) if positions[index] == 'GK']
        defenderPos = ["DF", "SW", "RWB", "LWB", "RB", "LB", "CB", "LCB", "RCB"]
        self.defenders = [index for index in range(len(positions)) if positions[index] in defenderPos]
        midPos = ["CAM", "CDM","LAM", "RAM", "MF", "DM", "LDM", "RDM", "LM", "RM", "CM", "LCM", "RCM","AM"]
        self.midfielders = [index for index in range(len(positions)) if positions[index] in midPos]
        forwardsPos = ["FW", "CF", "RF", "LF", "ST", "LS", "RS", "LW", "RW"]
        self.forwards = [index for index in range(len(positions)) if positions[index] in forwardsPos]

    def findWingPositions(self):
        allPositions = list(self.data["Position"])
        clubs = list(self.data["Club"])
        positions = [allPositions[index] for index in range(len(clubs)) if clubs[index] in self.teams]
        self.positions = positions
        self.goalkeepers = [index for index in range(len(positions)) if positions[index] == 'GK']
        wingPos = ["RWB", "LWB", "LDM", "RDM", "RF", "LF", "LAM", "RAM", "LS", "RS", "RW", "LW", "LF", "RF", "LCB", "RCB", "LB", "RB"]
        self.wings = [index for index in range(len(positions)) if positions[index] in wingPos]
        centerPos = ["SW", "CB", "LCB", "RCB", "CDM", "CAM", "DM", "AM", "ST", "CF"]
        self.center = [index for index in range(len(positions)) if positions[index] in centerPos]


    def makePlayersList(self, attributesList):
        players = []
        attributes = [list(self.data[attribute]) for attribute in attributesList]
        totalSkill = self.totalSkill()
        for i in self.indexes:
            player = [(attribute[i] / totalSkill[i]) for attribute in attributes]
            if not np.isnan(player).any():
                players.append(player)
        return players

    def totalSkill(self):
        skills = []
        attributes = self.getAllAttributes()
        for index in range(len(list(self.data["ID"]))):
            sum = 0
            for attribute in attributes:
                sum += attribute[index]
            skills.append(sum)
        return skills

    def getAllAttributes(self):

        allAttributeNames = ["Crossing", "Finishing", "HeadingAccuracy", "ShortPassing", "Volleys",
        "Dribbling", "Curve", "FKAccuracy","LongPassing", "BallControl", "Acceleration",
        "SprintSpeed", "Agility", "Reactions", "Balance", "ShotPower", "Jumping", "Stamina", "Strength",
        "LongShots", "Aggression", "Interceptions", "Positioning", "Vision", "Penalties", "Composure", "Marking",
        "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling", "GKKicking", "GKPositioning", "GKReflexes"]

        allAttributes = [list(self.data[attribute]) for attribute in allAttributeNames]

        return allAttributes