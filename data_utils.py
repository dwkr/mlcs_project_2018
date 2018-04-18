
# Core modules
import csv
import numpy as np
from datetime import datetime

pathToData = "/Users/tejallotlikar/Downloads/data/"


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
        if id == StadiumList[i][0]:
            #Debug print
            #print("found stadium with id",id)
            return StadiumList[i]
        
    print("Error getting Stadium info")
    return -1

#Prepares a dictionary(for next game) to add in list of dictionaries
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
    stadiumInfo = getStadiumInfoByID(singleGame[4], Stadium)
    newGame['Capacity'] = stadiumInfo[4]
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
    
    #Skip first 100 games
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
        if(game['HTStats'][35] >= game['VTStats'][35]):
            NumberOfWins[htindex] = NumberOfWins[htindex] + 1
        else:
            NumberOfWins[vtindex] = NumberOfWins[vtindex] + 1       
            
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
        
        HT_time_of_poss = game['HTStats'][58]  
        VT_time_of_poss = game['VTStats'][58]
        HT_penalty = game['HTStats'][59]
        VT_penalty = game['VTStats'][59]
        HT_Kickoff_yard = game['HTStats'][39]
        VT_Kickoff_yard = game['VTStats'][39]
        temp.extend((HT_time_of_poss,VT_time_of_poss,HT_penalty, VT_penalty, HT_Kickoff_yard, VT_Kickoff_yard))
        
        X_train.append(temp)
        if(game['HTStats'][35] >= game['VTStats'][35]):
            temp_y.append(1)#Y_train.append(1)
            NumberOfWins[htindex] = NumberOfWins[htindex] + 1
        else:
            temp_y.append(0)#Y_train.append(0)
            NumberOfWins[vtindex] = NumberOfWins[vtindex] + 1
        point_difference = game['HTStats'][35] - game['VTStats'][35]
        temp_y.append(point_difference)
        Y_train.append(temp_y)
        PointDifference[htindex] += point_difference
        PointDifference[vtindex] -= point_difference 
        NumberOfMatches[vtindex] = NumberOfMatches[vtindex] + 1
        NumberOfMatches[htindex] = NumberOfMatches[htindex] + 1
    print("X_train size: ", len(X_train))
    print("Y_train size: ", len(Y_train))

    return X_train, Y_train 


#Creates data for a specific season
def createData(SeasonList):
    X_train = []
    Y_train = []
    
    for year in SeasonList:
        x, y = getFeatures(year)
        X_train = X_train + x
        Y_train = Y_train + y
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    return X_train, Y_train

