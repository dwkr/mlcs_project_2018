
# coding: utf-8

# In[1]:


# Core modules
import csv
import numpy as np
from datetime import datetime

pathToData = "data/"
# MACROS
GMEGAMECODE = 0 # Game Game Code
GMEDATE = 1 # Game Date
GMEVISITTEAMCODE = 2 # Game Visit Team Code
GMEHOMETEAMCODE = 3 # Game Home Team Code
GMESTADIUMCODE = 4 # Game Stadium Code
GMSGAMECODE = 0 # Game Stats Game Code
GMSATTENDANCE = 1 # Game Stats Attendance
GMSDURATION = 2 # Game Stats Duration
STDSTADIUMCODE = 0 # Stadium Stadium Code
STDCAPACITY = 4 # Stadium Capacity
TGSPOINTS = 35 # Team Game Stats Points
TGSTIMEOFPOSS = 58 # Team Game Stats Time of Possesion
TGSPENALTY = 59 # Team Game Stats Penalty
TGSKICKOFFYARD = 39 # Team Game Stats Kick off yard
TGSFUMBLE = 43 #Team Game Stats Fumbles
TGSRUSHYARDS = 3 #Team Game Stats Rush Yards
TGSRUSHATTEMPTS = 2 ##Team Game Rush Attempts
TGSPASSYARDS = 7 #Team Game Stats Pass Yards
TGSPASSATTEMPTS = 5 ##Team Game Pass Attempts
TGSGOALMADE = 27 #Team Goals Made
TGSGOALATTEMPTS = 26 #Team Goal Attempts


def try_parsing_date(text):
    for fmt in ('%m-%d-%Y', '%m.%d.%Y', '%m/%d/%Y'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')
    

#Reads data from all files for a particular season
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
        
    return (Game,GameStatistics,TeamGameStatistics, Stadium)



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
   
#Gives Stadium Info stats based on the ID from the stadium list
def getStadiumInfoByID(id, StadiumList):
    
    for i in range(len(StadiumList)):
        if id == StadiumList[i][STDSTADIUMCODE]:
            #Debug print
            #print("found stadium with id",id)
            return StadiumList[i]
        
    print("Error getting Stadium info")
    return -1

#Prepares a dictionary(for next game) to add in list of dictionaries
def getNextGameStats(singleGame, singleGameStatistics, TeamGameStatistics, Stadium):
    newGame = {}
    #newGame['Date'] = try_parsing_date(singleGame[1])
    newGame['Date'] = singleGame[GMEDATE]
    newGame['GameID'] = singleGame[GMEGAMECODE]
    newGame['Attendance'] = singleGameStatistics[GMSATTENDANCE]
    #print(GameStatistics[2])
    newGame['Duration'] = singleGameStatistics[GMSDURATION]
    newGame['HomeTeam'] = int(singleGame[GMEHOMETEAMCODE])
    newGame['VisitTeam'] = int(singleGame[GMEVISITTEAMCODE])
    HomeTeamStats, VisitTeamStats = getTeamGameStatsByID(singleGame[GMEGAMECODE],TeamGameStatistics,singleGame[GMEHOMETEAMCODE],singleGame[GMEVISITTEAMCODE])
    #newGame['HTStats'] = HomeTeamStats
    #newGame['VTStats'] = VisitTeamStats
    newGame['HTStats'] = list(map(float,HomeTeamStats))
    newGame['VTStats'] = list(map(float,VisitTeamStats))
    stadiumInfo = getStadiumInfoByID(singleGame[GMESTADIUMCODE], Stadium)
    newGame['Capacity'] = stadiumInfo[STDCAPACITY]
    #GameList.append(newGame)
    return newGame

	
	
#Prepares gamelist and teamlist for a specified year
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
    print("GameList size: ", len(GameList))
    print("TeamList size: ", len(TeamList))

    return GameList, TeamList


	
#Generates X_data and Y_data for a specific year
def getFeatures(year):
    
    GameList, TeamList = prepareGameAndTeamList(year)
        #Initializing for each team
    NumberOfWins = np.zeros(len(TeamList))
    NumberOfMatches = np.zeros(len(TeamList))
    PointDifference = np.zeros(len(TeamList))
    Indicator = np.zeros(len(TeamList))
    TimeOfPossession = np.zeros(len(TeamList))
    Penalty = np.zeros(len(TeamList))
    KickoffYard = np.zeros(len(TeamList))
    indicator = np.zeros(len(TeamList))
    Fumbles = np.zeros(len(TeamList))
    RushYards = np.zeros(len(TeamList))
    RushAttempts = np.zeros(len(TeamList))
    RushRatio = np.zeros(len(TeamList))
    PassYards = np.zeros(len(TeamList))
    PassAttempts = np.zeros(len(TeamList))
    PassRatio = np.zeros(len(TeamList))
    Attendance = np.zeros(len(TeamList))
    Duration = np.zeros(len(TeamList))
    TeamGoalsMade = np.zeros(len(TeamList))
    TeamGoalsAttempts = np.zeros(len(TeamList))
    TeamGoalRatio = np.zeros(len(TeamList))
    FirstDown = np.zeros(len(TeamList))

    
    #Skip first 100 games
    gamecount = 0
    for i,game in enumerate(GameList[:100]):
        gamecount = gamecount + 1
        htindex = TeamList.index(game['HomeTeam'])
        vtindex = TeamList.index(game['VisitTeam'])
        NumberOfMatches[htindex] = NumberOfMatches[htindex] + 1
        NumberOfMatches[vtindex] = NumberOfMatches[vtindex] + 1
        point_difference = game['HTStats'][TGSPOINTS] - game['VTStats'][TGSPOINTS]
        #game['HTStats'][35]  ----> points of home team
        Indicator[htindex] += game['HTStats'][TGSPOINTS]
        Indicator[vtindex] += game['VTStats'][TGSPOINTS]
        PointDifference[htindex] += point_difference
        PointDifference[vtindex] -= point_difference 
        if(game['HTStats'][TGSPOINTS] >= game['VTStats'][TGSPOINTS]):
            NumberOfWins[htindex] = NumberOfWins[htindex] + 1
        else:
            NumberOfWins[vtindex] = NumberOfWins[vtindex] + 1 
        TimeOfPossession[htindex] += game['HTStats'][TGSTIMEOFPOSS]  
        TimeOfPossession[vtindex] += game['VTStats'][TGSTIMEOFPOSS]
        Penalty[htindex] += game['HTStats'][TGSPENALTY]
        Penalty[vtindex] += game['VTStats'][TGSPENALTY]
        KickoffYard[htindex] += game['HTStats'][TGSKICKOFFYARD]
        KickoffYard[vtindex] += game['VTStats'][TGSKICKOFFYARD]
        indicator = Indicator[htindex]/(1 + NumberOfMatches[htindex]) - Indicator[vtindex]/(1 + NumberOfMatches[vtindex])
        Fumbles[htindex] += game['HTStats'][TGSFUMBLE]
        Fumbles[vtindex] += game['VTStats'][TGSFUMBLE]
        RushYards[htindex] += game['HTStats'][TGSRUSHYARDS]
        RushYards[vtindex] += game['VTStats'][TGSRUSHYARDS]
        RushAttempts[htindex] += game['HTStats'][TGSRUSHATTEMPTS]
        RushAttempts[vtindex] += game['VTStats'][TGSRUSHATTEMPTS]
        RushRatio[htindex] += RushYards[htindex]/ RushAttempts[htindex]
        RushRatio[htindex] += RushYards[vtindex]/ RushAttempts[vtindex]
        PassYards[htindex] += game['HTStats'][TGSPASSYARDS]
        PassYards[vtindex] += game['VTStats'][TGSPASSYARDS]
        PassAttempts[htindex] += game['HTStats'][TGSPASSATTEMPTS]
        PassAttempts[vtindex] += game['VTStats'][TGSPASSATTEMPTS]
        PassRatio[htindex] += PassYards[htindex]/ PassAttempts[htindex]
        PassRatio[htindex] += PassYards[vtindex]/ PassAttempts[vtindex]
        TeamGoalsMade[htindex] += game['HTStats'][TGSGOALMADE]+1
        TeamGoalsAttempts[htindex] += game['HTStats'][TGSGOALATTEMPTS]+1
        TeamGoalRatio[htindex] += TeamGoalsMade[htindex]/TeamGoalsAttempts[htindex]
        TeamGoalsMade[vtindex] += game['VTStats'][TGSGOALMADE]+1
        TeamGoalsAttempts[vtindex] += game['VTStats'][TGSGOALATTEMPTS]+1
        TeamGoalRatio[vtindex] += TeamGoalsMade[vtindex]/TeamGoalsAttempts[vtindex]
        
        #Attendance[htindex] += game['Attendance']
        #Attendance[vtindex] += game['Attendance']
        #Duration[htindex] += game['Duration']
        #Duration[vtindex] += game['Duration']
            
    #win ratios of home team, win ratio of visit team
    X_train = []
    X_test = []
    #1 if home team wins, zero otherwise
    Y_train = []
    Y_test = []
    
    #debug print
    for i,team in enumerate(TeamList):
        if(NumberOfMatches[i] < NumberOfWins[i]):
            print("Matches: ", NumberOfMatches[i], " Wins: ", NumberOfWins[i])
        
    gamecount = 0
    for i,game in enumerate(GameList[100:]):
        temp = []
        temp_y = []
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
        
        Indicator[htindex] += game['HTStats'][TGSPOINTS]
        Indicator[vtindex] += game['VTStats'][TGSPOINTS]
        #HT_time_of_poss[htindex] += ['HTStats'][TGSTIMEOFPOSS]  
        #print("HT_time",HT_time_of_poss)
        #VT_time_of_poss[vtindex] += game['VTStats'][TGSTIMEOFPOSS]
        TimeOfPossession[htindex] += game['HTStats'][TGSTIMEOFPOSS]  
        TimeOfPossession[vtindex] += game['VTStats'][TGSTIMEOFPOSS]
        Penalty[htindex] += game['HTStats'][TGSPENALTY]
        Penalty[vtindex] += game['VTStats'][TGSPENALTY]
        KickoffYard[htindex] += game['HTStats'][TGSKICKOFFYARD]
        KickoffYard[vtindex] += game['VTStats'][TGSKICKOFFYARD]
        indicator = Indicator[htindex]/(1 + NumberOfMatches[htindex]) - Indicator[vtindex]/(1 + NumberOfMatches[vtindex])
        Fumbles[htindex] += game['HTStats'][TGSFUMBLE]
        Fumbles[vtindex] += game['VTStats'][TGSFUMBLE]
        RushYards[htindex] += game['HTStats'][TGSRUSHYARDS]
        RushYards[vtindex] += game['VTStats'][TGSRUSHYARDS]
        RushAttempts[htindex] += game['HTStats'][TGSRUSHATTEMPTS]
        RushAttempts[vtindex] += game['VTStats'][TGSRUSHATTEMPTS]
        RushRatio[htindex] += RushYards[htindex]/ RushAttempts[htindex]
        RushRatio[htindex] += RushYards[vtindex]/ RushAttempts[vtindex]
        PassYards[htindex] += game['HTStats'][TGSPASSYARDS]
        PassYards[vtindex] += game['VTStats'][TGSPASSYARDS]
        PassAttempts[htindex] += game['HTStats'][TGSPASSATTEMPTS]
        PassAttempts[vtindex] += game['VTStats'][TGSPASSATTEMPTS]
        PassRatio[htindex] += PassYards[htindex]/ PassAttempts[htindex]
        PassRatio[htindex] += PassYards[vtindex]/ PassAttempts[vtindex]
        TeamGoalsMade[htindex] += game['HTStats'][TGSGOALMADE]+1
        TeamGoalsAttempts[htindex] += game['HTStats'][TGSGOALATTEMPTS]+1
        TeamGoalRatio[htindex] += TeamGoalsMade[htindex]/TeamGoalsAttempts[htindex]
        TeamGoalsMade[vtindex] += game['VTStats'][TGSGOALMADE]+1
        TeamGoalsAttempts[vtindex] += game['VTStats'][TGSGOALATTEMPTS]+1
        TeamGoalRatio[vtindex] += TeamGoalsMade[vtindex]/TeamGoalsAttempts[vtindex]
        
        #Attendance[htindex] += game['Attendance']
        #Attendance[vtindex] += game['Attendance']
        #Duration[htindex] += game['Duration']
        #Duration[vtindex] += game['Duration']
        
        temp.extend((TimeOfPossession[htindex]/(1 + NumberOfMatches[htindex]),TimeOfPossession[vtindex]/(1 + NumberOfMatches[vtindex]),
                     Penalty[htindex]/(1 + NumberOfMatches[htindex]),  Penalty[vtindex]/(1 + NumberOfMatches[vtindex]), 
                     KickoffYard[htindex]/(1 + NumberOfMatches[htindex]), KickoffYard[vtindex]/(1 + NumberOfMatches[vtindex]), 
                     indicator,
                     Fumbles[htindex]/(1 + NumberOfMatches[htindex]),Fumbles[vtindex]/(1 + NumberOfMatches[vtindex]), 
                     RushRatio[htindex]/(1 + NumberOfMatches[htindex]) , RushRatio[vtindex]/(1 + NumberOfMatches[vtindex]) , 
                     PassRatio[htindex]/(1 + NumberOfMatches[htindex]) , PassRatio[vtindex]/(1 + NumberOfMatches[vtindex]),
                     TeamGoalRatio[htindex]/(1 + NumberOfMatches[htindex]) , TeamGoalRatio[vtindex]/(1 + NumberOfMatches[vtindex]) 
                     #attendance, 
                     #duration 
                    ))
        
        X_train.append(temp)
        if(game['HTStats'][TGSPOINTS] >= game['VTStats'][TGSPOINTS]):
            temp_y.append(1)#Y_train.append(1)
            NumberOfWins[htindex] = NumberOfWins[htindex] + 1
        else:
            temp_y.append(0)#Y_train.append(0)
            NumberOfWins[vtindex] = NumberOfWins[vtindex] + 1
        point_difference = game['HTStats'][TGSPOINTS] - game['VTStats'][TGSPOINTS]
        temp_y.append(point_difference)
        Y_train.append(temp_y)
        PointDifference[htindex] += point_difference
        PointDifference[vtindex] -= point_difference 
        NumberOfMatches[vtindex] = NumberOfMatches[vtindex] + 1
        NumberOfMatches[htindex] = NumberOfMatches[htindex] + 1
    print("X_train size: ", len(X_train))
    #print("X_train shape: ", X_train.shape)
    print("Y_train size: ", len(Y_train))

    return X_train, Y_train 


#Creates data for a specific season
def createData(SeasonList, path):
    X_train = []
    Y_train = []
    pathToData = path
    for year in SeasonList:
        x, y = getFeatures(year)
        X_train = X_train + x
        Y_train = Y_train + y
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    return X_train, Y_train


# In[2]:


#from data_utils import *

SeasonList = [2005, 2006,2007,2008,2009,2010,2011, 2012,2013]

#X_TRAIN, Y_TRAIN = createTrainingData(SeasonList)
#X_DATA, Y_DATA = createTrainingData(SeasonList)
X_DATA, Y_DATA = createData(SeasonList,"data/")


X_DATA


Y_DATA


print("XDATA length" ,len(X_DATA))

X_DATA[1]

type(X_DATA)

splitindex = (int)(0.7*len(X_DATA))
X_TRAIN = X_DATA[:splitindex]
X_TEST= X_DATA[splitindex:]
Y_TRAIN = Y_DATA[:splitindex]
Y_TEST= Y_DATA[splitindex:]

print (len(X_TRAIN))
print (len(X_TEST))
print (len(Y_TRAIN))
print (len(Y_TEST))

X_TRAIN[:, :2]

X_TRAIN[2]




# In[3]:



X_TRAIN[1]



# In[78]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 20000, 'max_depth': 10,
        'learning_rate': 0.05, 'loss': 'ls'}
clf = GradientBoostingRegressor(**params).fit(X_TRAIN[:,0:4], Y_TRAIN[:,1])

# For me, the Mean Squared Error wasn't much informative and used instead the
# :math:`R^2` **coefficient of determination**. This measure is a number
# indicating how well a variable is able to predict the other. Numbers close to
# 0 means poor prediction and numbers close to 1 means perfect prediction. In the
# book, they claim a 0.84 against a 0.86 reported in the paper that created the
# dataset using a highly tuned algorithm. I'm getting a good 0.83 without much
# tunning of the parameters so it's a good out of the box technique.

print(clf.predict(X_TEST[:,0:4]))
print(Y_TEST[:,1])
#accuracyTest = accuracy_score(Y_TEST[:,1], clf.predict(X_TEST))
#accuracyTest = clf.accuracy_score_(Y_TEST[:,1], clf.predict(X_TEST))
mse = mean_squared_error(Y_TEST[:,1], clf.predict(X_TEST[:,0:4]))
r2 = r2_score(Y_TEST[:,1], clf.predict(X_TEST[:,0:4]))
loss = clf.loss_(Y_TEST[:,1], clf.predict(X_TEST[:,0:4]))
#print("Accuracy:", accuracyTest)

print("loss", loss)

print("MSE: %.4f" % mse)
print("R2: %.4f" % r2)


# In[79]:


def Accuracy(Y_label,Y_hat, epsilon):
    CorrectPredictions = 0
    print(len(Y_label))
    print(len(Y_hat))
    for i,current in enumerate(Y_label):
        if(abs(Y_label[i] - Y_hat[i]) <= epsilon):
            CorrectPredictions = CorrectPredictions + 1
    return(100 * CorrectPredictions/len(Y_label), CorrectPredictions)


# In[85]:


accuracy, correct_predictions = Accuracy(Y_TEST[:,1] , clf.predict(X_TEST[:,0:4]), 20)


# In[86]:


accuracy


# In[82]:


correct_predictions

