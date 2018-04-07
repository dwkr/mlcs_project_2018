
# coding: utf-8

# In[1]:


import csv
import numpy as np
from sklearn.metrics import log_loss
import torch
from torch import autograd,nn, optim
import torch.nn.functional as F
from datetime import datetime

pathToData = "data/"


# In[2]:


def try_parsing_date(text):
    for fmt in ('%m-%d-%Y', '%m.%d.%Y', '%m/%d/%Y'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')
    
#with open('data-2005/team-game-statistics.csv', 'r') as f:
 # reader = csv.reader(f)
  #TeamGameStatistics = list(reader)


# In[3]:


def ReadDataForASeason(year):
    with open(pathToData + str(year) +'/game.csv', 'r') as f:
        reader = csv.reader(f)
        Game = list(reader)
        Game = Game[1:]

    with open(pathToData + str(year) + '/game-statistics.csv', 'r') as f:
        reader = csv.reader(f)
        GameStatistics = list(reader)
        GameStatistics = GameStatistics[1:]
    
    with open(pathToData + str(year) + '/team-game-statistics.csv', 'r') as f:
        reader = csv.reader(f)
        TeamGameStatistics = list(reader)
        TeamGameStatistics = TeamGameStatistics[1:]
        
    with open(pathToData + str(year) + '/conference.csv', 'r') as f:
        reader = csv.reader(f)
        Conference = list(reader)
        Conference = Conference[1:]
        
    with open(pathToData + str(year) + '/drive.csv', 'r') as f:
        reader = csv.reader(f)
        Drive = list(reader)
        Drive = Drive[1:]
                
    with open(pathToData + str(year) + '/kickoff.csv', 'r') as f:
        reader = csv.reader(f)
        Kickoff = list(reader)
        Kickoff = Kickoff[1:]
                
    with open(pathToData + str(year) + '/kickoff-return.csv', 'r') as f:
        reader = csv.reader(f)
        KickoffReturn = list(reader)
        KickoffReturn = KickoffReturn[1:]
                
    with open(pathToData + str(year) + '/pass.csv', 'r') as f:
        reader = csv.reader(f)
        Pass = list(reader)
        Pass = Pass[1:]
                
    with open(pathToData + str(year) + '/play.csv', 'r') as f:
        reader = csv.reader(f)
        Play = list(reader)
        Play = Play[1:]
                
    with open(pathToData + str(year) + '/player.csv', 'r') as f:
        reader = csv.reader(f)
        Player = list(reader)
        Player = Player[1:]
                
    with open(pathToData + str(year) + '/player-game-statistics.csv', 'r') as f:
        reader = csv.reader(f)
        PlayerGameStatistics = list(reader)
        PlayerGameStatistics = PlayerGameStatistics[1:]
                
    with open(pathToData + str(year) + '/punt.csv', 'r') as f:
        reader = csv.reader(f)
        Punt = list(reader)
        Punt = Punt[1:]
                
    with open(pathToData + str(year) + '/punt-return.csv', 'r') as f:
        reader = csv.reader(f)
        PuntReturn = list(reader)
        PuntReturn = PuntReturn[1:]
                
    with open(pathToData + str(year) + '/reception.csv', 'r') as f:
        reader = csv.reader(f)
        Reception = list(reader)
        Reception = Reception[1:]
                
    with open(pathToData + str(year) + '/rush.csv', 'r') as f:
        reader = csv.reader(f)
        Rush = list(reader)
        Rush = Rush[1:]
                
    with open(pathToData + str(year) + '/stadium.csv', 'r') as f:
        reader = csv.reader(f)
        Stadium = list(reader)
        Stadium = Stadium[1:]
        Stadium = sorted(Stadium, key=lambda x: float(x[0]))
                
    with open(pathToData + str(year) + '/team.csv', 'r') as f:
        reader = csv.reader(f)
        Team = list(reader)
        Team = Team[1:]
        
    return (Game,GameStatistics,TeamGameStatistics,Stadium)


# In[4]:


#print(Game[100][1])
#date1 = try_parsing_date(Game[2][1])
#date2 = try_parsing_date(Game[100][1])

#if date1 < date2:
#    print(date1)

#print(date2)

#for i in range(len(Game)):
#    Game[i][1] = try_parsing_date(Game[i][1])

#Game[150][1]


# In[5]:


#Assuming no anomolies in game data in TEAMGAMESTATS correspnoding to games in GAME
#Add fields from TEAMSGAMESTATS in this function
def getTeamGameStatsByID(GameID, TeamGameStatistics, HomeTeam, VisitTeam):
    count = 0
    for i,gamestats in enumerate(TeamGameStatistics):
        if(gamestats[1] == GameID):
            if(gamestats[0] == HomeTeam):
                HomeTeamStats = gamestats
            else:
                VisitTeamStats = gamestats
            count = count + 1
        if(count == 2):
            return(HomeTeamStats,VisitTeamStats)
    print("Loop ended without returning! :( ")


# In[6]:


def getNextGameStats(singleGame, singleGameStatistics, TeamGameStatistics, Stadium):
    newGame = {}
    #newGame['Date'] = try_parsing_date(singleGame[1])
    newGame['Date'] = singleGame[1]
    newGame['GameID'] = singleGame[0]
    newGame['Attendance'] = singleGameStatistics[1]
    #print(GameStatistics[2])
    newGame['Duration'] = singleGameStatistics[2]
    newGame['HomeTeam'] = int(singleGame[3])
    newGame['VisitTeam'] = int(singleGame[2])
    HomeTeamStats, VisitTeamStats = getTeamGameStatsByID(singleGame[0],TeamGameStatistics,singleGame[3],singleGame[2])
    #newGame['HTStats'] = HomeTeamStats
    #newGame['VTStats'] = VisitTeamStats
    newGame['HTStats'] = list(map(float,HomeTeamStats))
    newGame['VTStats'] = list(map(float,VisitTeamStats))
    newGame['Capacity'] = Stadium[int(singleGame[4])-1][4]
    #GameList.append(newGame)
    return newGame


# In[7]:


def prepareGameAndTeamList(year):
    Game, GameStatistics, TeamGameStatistics, Stadium = ReadDataForASeason(year)
    GameList = [] #List of dictionaries, based on sorted order of games
    TeamList = [] #List of Teams in sorted order of Team codes
    count = 0
    for i in range(len(Game)):
        GameList.append(getNextGameStats(Game[i], GameStatistics[i], TeamGameStatistics,Stadium))
        #GameList
        if(GameList[i]['HomeTeam'] in TeamList):
            count = 1
        else:
            TeamList.append(GameList[i]['HomeTeam'])
        if(GameList[i]['VisitTeam'] in TeamList):
            count = 1
        else:
            TeamList.append(GameList[i]['VisitTeam'])
        
    TeamList.sort()
    return GameList, TeamList


# In[8]:


def getFeatures(year):
    
    GameList, TeamList = prepareGameAndTeamList(year)
        #Initializing for each team
    NumberOfWins = np.zeros(len(TeamList))
    NumberOfMatches = np.zeros(len(TeamList))
    PointDifference = np.zeros(len(TeamList))
    
    #Skip first 100 games
    #TODO: Change 100 games to two weeks
    gamecount = 0
    for i,game in enumerate(GameList[:100]):
        gamecount = gamecount + 1
        htindex = TeamList.index(game['HomeTeam'])
        vtindex = TeamList.index(game['VisitTeam'])
        NumberOfMatches[htindex] = NumberOfMatches[htindex] + 1
        NumberOfMatches[vtindex] = NumberOfMatches[vtindex] + 1
        point_difference = game['HTStats'][35] - game['VTStats'][35]
        PointDifference[htindex] += point_difference
        PointDifference[vtindex] -= point_difference 
        #TODO: Check condition for draw
        if(point_difference>=0):
            NumberOfWins[htindex] = NumberOfWins[htindex] + 1
        else:
            NumberOfWins[vtindex] = NumberOfWins[vtindex] + 1       
            
    #win ratios of home team, win ratio of visit team
    X_train = []
    #1 if home team wins, zero otherwise
    Y_train = []
    
    #debug print
    for i,team in enumerate(TeamList):
        if(NumberOfMatches[i] < NumberOfWins[i]):
            print("Matches: ", NumberOfMatches[i], " Wins: ", NumberOfWins[i])
        
    gamecount = 0
    for i,game in enumerate(GameList[100:]):
        temp = []
        gamecount = gamecount + 1
        #print("For ",i,": ")
        #print(game)
        htindex = TeamList.index(game['HomeTeam'])
        vtindex = TeamList.index(game['VisitTeam'])
        #print(NumberOfWins[htindex])
        #print(NumberOfMatches[htindex])
        #print(NumberOfWins[vtindex])
        #print(NumberOfMatches[vtindex])
        if(NumberOfMatches[htindex] > 0):
            temp.append(NumberOfWins[htindex]/NumberOfMatches[htindex])
        else:
            temp.append(0.0)
        if(NumberOfMatches[vtindex] > 0):
            temp.append(NumberOfWins[vtindex]/NumberOfMatches[vtindex])
        else:
            temp.append(0.0)
        temp.append(PointDifference[htindex])
        temp.append(PointDifference[vtindex])
        X_train.append(temp)
        point_difference = game['HTStats'][35] - game['VTStats'][35]
        # TODO: check for draw
        if(point_difference>=0):
            Y_train.append(1)
            NumberOfWins[htindex] = NumberOfWins[htindex] + 1
        else:
            Y_train.append(0)
            NumberOfWins[vtindex] = NumberOfWins[vtindex] + 1
        PointDifference[htindex] += point_difference
        PointDifference[vtindex] -= point_difference 
        NumberOfMatches[vtindex] = NumberOfMatches[vtindex] + 1
        NumberOfMatches[htindex] = NumberOfMatches[htindex] + 1
        
    return X_train, Y_train 


# In[9]:


def createData(SeasonList):
    X_train = []
    Y_train = []
    
    for year in SeasonList:
        x, y = getFeatures(year)
        X_train += x
        Y_train += y
        
    #print(x)
        
    return np.array(X_train), np.array(Y_train)

