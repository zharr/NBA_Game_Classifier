import os

import pandas as pd

RAW_DATA = '~/Documents/NBA_Game_Classifier/Data/Raw_Data'

df_all = pd.read_csv(os.path.join(RAW_DATA, 'games_2018_2019.csv'))

# strucutre of feature sdata
# date, away team avg stats, home team avg stats
#
# structure of class data
# away win {0,1}, home win (!away_win)

# need to come up with a way to do this...
# thinking 2D dict with team name, date as key

team_avgs = {
    'Atlanta Hawks': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Boston Celtics': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Brooklyn Nets': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Charlotte Hornets': {         #Bobcats': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Chicago Bulls': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Cleveland Cavaliers': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Dallas Mavericks': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Denver Nuggets': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Detroit Pistons': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Golden State Warriors': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Houston Rockets': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Indiana Pacers': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Los Angeles Clippers': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Los Angeles Lakers': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Memphis Grizzlies': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Miami Heat': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Milwaukee Bucks': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Minnesota Timberwolves': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'New Orleans Pelicans': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'New York Knicks': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Oklahoma City Thunder': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Orlando Magic': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Philadelphia 76ers': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Phoenix Suns': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Portland Trail Blazers': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Sacramento Kings': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'San Antonio Spurs': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Toronto Raptors': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Utah Jazz': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    },
    'Washington Wizards': {
        'fg': {},
        'fg%': {},
        '3p': {},
        '3p%': {},
        'ft': {},
        'ft%': {},
        'orb': {},
        'drb': {},
        'ast': {},
        'stl': {},
        'blk': {},
        'tov': {},
        'pts': {}
    }
}

for team, date in team_avgs.items():
    team_avgs[team]['fg'][0] = [0, 0]
    team_avgs[team]['fg%'][0] = [0,0]
    team_avgs[team]['3p'][0] = [0, 0]
    team_avgs[team]['3p%'][0] = [0,0]
    team_avgs[team]['ft'][0] = [0, 0]
    team_avgs[team]['ft%'][0] = [0,0]
    team_avgs[team]['orb'][0] = [0,0]
    team_avgs[team]['drb'][0] = [0,0]
    team_avgs[team]['ast'][0] = [0,0]
    team_avgs[team]['stl'][0] = [0,0]
    team_avgs[team]['blk'][0] = [0,0]
    team_avgs[team]['tov'][0] = [0,0]
    team_avgs[team]['pts'][0] = [0,0]

# need to first calculate averages before of every game
for index, row in df_all.iterrows():
    date = row['Date']
    away_team = row['Away Team']
    home_team = row['Home Team']
    away_stats = []
    home_stats = []
    winner = [-1,-1]

    # field goals
    if False:
        fg_a = row['FG_A']
        fg_h = row['FG_H']
        team_avgs[away_team]['fg'][date] = team_avgs[away_team]['fg'][0][1]
        away_stats.append(team_avgs[away_team]['fg'][0][1])
        team_avgs[away_team]['fg'][0][0] += 1
        team_avgs[away_team]['fg'][0][1] = ((team_avgs[away_team]['fg'][0][1] * (
        team_avgs[away_team]['fg'][0][0] - 1)) + fg_a) / team_avgs[away_team]['fg'][0][0]
        team_avgs[home_team]['fg'][date] = team_avgs[home_team]['fg'][0][1]
        home_stats.append(team_avgs[home_team]['fg'][0][1])
        team_avgs[home_team]['fg'][0][0] += 1
        team_avgs[home_team]['fg'][0][1] = ((team_avgs[home_team]['fg'][0][1] * (
        team_avgs[home_team]['fg'][0][0] - 1)) + fg_h) / team_avgs[home_team]['fg'][0][0]

    # field goal percentage
    fgp_a = row['FG%_A']
    fgp_h = row['FG%_H']
    team_avgs[away_team]['fg%'][date] = team_avgs[away_team]['fg%'][0][1]
    away_stats.append(team_avgs[away_team]['fg%'][0][1])
    team_avgs[away_team]['fg%'][0][0] += 1
    team_avgs[away_team]['fg%'][0][1] = ((team_avgs[away_team]['fg%'][0][1]*(team_avgs[away_team]['fg%'][0][0]-1)) + fgp_a) / team_avgs[away_team]['fg%'][0][0]
    team_avgs[home_team]['fg%'][date] = team_avgs[home_team]['fg%'][0][1]
    home_stats.append(team_avgs[home_team]['fg%'][0][1])
    team_avgs[home_team]['fg%'][0][0] += 1
    team_avgs[home_team]['fg%'][0][1] = ((team_avgs[home_team]['fg%'][0][1]*(team_avgs[home_team]['fg%'][0][0]-1)) + fgp_h) / team_avgs[home_team]['fg%'][0][0]

    # three points
    if False:
        tp_a = row['3P_A']
        tp_h = row['3P_H']
        team_avgs[away_team]['3p'][date] = team_avgs[away_team]['3p'][0][1]
        away_stats.append(team_avgs[away_team]['3p'][0][1])
        team_avgs[away_team]['3p'][0][0] += 1
        team_avgs[away_team]['3p'][0][1] = ((team_avgs[away_team]['3p'][0][1] * (
        team_avgs[away_team]['3p'][0][0] - 1)) + tp_a) / team_avgs[away_team]['3p'][0][0]
        team_avgs[home_team]['3p'][date] = team_avgs[home_team]['3p'][0][1]
        home_stats.append(team_avgs[home_team]['3p'][0][1])
        team_avgs[home_team]['3p'][0][0] += 1
        team_avgs[home_team]['3p'][0][1] = ((team_avgs[home_team]['3p'][0][1] * (
        team_avgs[home_team]['3p'][0][0] - 1)) + tp_h) / team_avgs[home_team]['3p'][0][0]

    # three point percentage
    tpp_a = row['3P%_A']
    tpp_h = row['3P%_H']
    team_avgs[away_team]['3p%'][date] = team_avgs[away_team]['3p%'][0][1]
    away_stats.append(team_avgs[away_team]['3p%'][0][1])
    team_avgs[away_team]['3p%'][0][0] += 1
    team_avgs[away_team]['3p%'][0][1] = ((team_avgs[away_team]['3p%'][0][1] * (team_avgs[away_team]['3p%'][0][0] - 1)) + tpp_a) / team_avgs[away_team]['3p%'][0][0]
    team_avgs[home_team]['3p%'][date] = team_avgs[home_team]['3p%'][0][1]
    home_stats.append(team_avgs[home_team]['3p%'][0][1])
    team_avgs[home_team]['3p%'][0][0] += 1
    team_avgs[home_team]['3p%'][0][1] = ((team_avgs[home_team]['3p%'][0][1] * (team_avgs[home_team]['3p%'][0][0] - 1)) + tpp_h) / team_avgs[home_team]['3p%'][0][0]

    # free throws
    if False:
        ft_a = row['FT_A']
        ft_h = row['FT_H']
        team_avgs[away_team]['ft'][date] = team_avgs[away_team]['ft'][0][1]
        away_stats.append(team_avgs[away_team]['ft'][0][1])
        team_avgs[away_team]['ft'][0][0] += 1
        team_avgs[away_team]['ft'][0][1] = ((team_avgs[away_team]['ft'][0][1] * (
        team_avgs[away_team]['ft'][0][0] - 1)) + ft_a) / team_avgs[away_team]['ft'][0][0]
        team_avgs[home_team]['ft'][date] = team_avgs[home_team]['ft'][0][1]
        home_stats.append(team_avgs[home_team]['ft'][0][1])
        team_avgs[home_team]['ft'][0][0] += 1
        team_avgs[home_team]['ft'][0][1] = ((team_avgs[home_team]['ft'][0][1] * (
        team_avgs[home_team]['ft'][0][0] - 1)) + ft_h) / team_avgs[home_team]['ft'][0][0]

    # free throw percentage
    ftp_a = row['FT%_A']
    ftp_h = row['FT%_H']
    team_avgs[away_team]['ft%'][date] = team_avgs[away_team]['ft%'][0][1]
    away_stats.append(team_avgs[away_team]['ft%'][0][1])
    team_avgs[away_team]['ft%'][0][0] += 1
    team_avgs[away_team]['ft%'][0][1] = ((team_avgs[away_team]['ft%'][0][1] * (team_avgs[away_team]['ft%'][0][0] - 1)) + ftp_a) / team_avgs[away_team]['ft%'][0][0]
    team_avgs[home_team]['ft%'][date] = team_avgs[home_team]['ft%'][0][1]
    home_stats.append(team_avgs[home_team]['ft%'][0][1])
    team_avgs[home_team]['ft%'][0][0] += 1
    team_avgs[home_team]['ft%'][0][1] = ((team_avgs[home_team]['ft%'][0][1] * (team_avgs[home_team]['ft%'][0][0] - 1)) + ftp_h) / team_avgs[home_team]['ft%'][0][0]

    # offensive rebounds
    orb_a = row['ORB_A']
    orb_h = row['ORB_H']
    team_avgs[away_team]['orb'][date] = team_avgs[away_team]['orb'][0][1]
    away_stats.append(team_avgs[away_team]['orb'][0][1])
    team_avgs[away_team]['orb'][0][0] += 1
    team_avgs[away_team]['orb'][0][1] = ((team_avgs[away_team]['orb'][0][1] * (team_avgs[away_team]['orb'][0][0] - 1)) + orb_a) / team_avgs[away_team]['orb'][0][0]
    team_avgs[home_team]['orb'][date] = team_avgs[home_team]['orb'][0][1]
    home_stats.append(team_avgs[home_team]['orb'][0][1])
    team_avgs[home_team]['orb'][0][0] += 1
    team_avgs[home_team]['orb'][0][1] = ((team_avgs[home_team]['orb'][0][1] * (team_avgs[home_team]['orb'][0][0] - 1)) + orb_h) / team_avgs[home_team]['orb'][0][0]

    # defensive rebounds
    drb_a = row['DRB_A']
    drb_h = row['DRB_H']
    team_avgs[away_team]['drb'][date] = team_avgs[away_team]['drb'][0][1]
    away_stats.append(team_avgs[away_team]['drb'][0][1])
    team_avgs[away_team]['drb'][0][0] += 1
    team_avgs[away_team]['drb'][0][1] = ((team_avgs[away_team]['drb'][0][1] * (team_avgs[away_team]['drb'][0][0] - 1)) + drb_a) / team_avgs[away_team]['drb'][0][0]
    team_avgs[home_team]['drb'][date] = team_avgs[home_team]['drb'][0][1]
    home_stats.append(team_avgs[home_team]['drb'][0][1])
    team_avgs[home_team]['drb'][0][0] += 1
    team_avgs[home_team]['drb'][0][1] = ((team_avgs[home_team]['drb'][0][1] * (team_avgs[home_team]['drb'][0][0] - 1)) + drb_h) / team_avgs[home_team]['drb'][0][0]

    # assists
    ast_a = row['AST_A']
    ast_h = row['AST_H']
    team_avgs[away_team]['ast'][date] = team_avgs[away_team]['ast'][0][1]
    away_stats.append(team_avgs[away_team]['ast'][0][1])
    team_avgs[away_team]['ast'][0][0] += 1
    team_avgs[away_team]['ast'][0][1] = ((team_avgs[away_team]['ast'][0][1] * (team_avgs[away_team]['ast'][0][0] - 1)) + ast_a) / team_avgs[away_team]['ast'][0][0]
    team_avgs[home_team]['ast'][date] = team_avgs[home_team]['ast'][0][1]
    home_stats.append(team_avgs[home_team]['ast'][0][1])
    team_avgs[home_team]['ast'][0][0] += 1
    team_avgs[home_team]['ast'][0][1] = ((team_avgs[home_team]['ast'][0][1] * (team_avgs[home_team]['ast'][0][0] - 1)) + ast_h) / team_avgs[home_team]['ast'][0][0]

    # steals
    stl_a = row['STL_A']
    stl_h = row['STL_H']
    team_avgs[away_team]['stl'][date] = team_avgs[away_team]['stl'][0][1]
    away_stats.append(team_avgs[away_team]['stl'][0][1])
    team_avgs[away_team]['stl'][0][0] += 1
    team_avgs[away_team]['stl'][0][1] = ((team_avgs[away_team]['stl'][0][1] * (team_avgs[away_team]['stl'][0][0] - 1)) + stl_a) / team_avgs[away_team]['stl'][0][0]
    team_avgs[home_team]['stl'][date] = team_avgs[home_team]['stl'][0][1]
    home_stats.append(team_avgs[home_team]['stl'][0][1])
    team_avgs[home_team]['stl'][0][0] += 1
    team_avgs[home_team]['stl'][0][1] = ((team_avgs[home_team]['stl'][0][1] * (team_avgs[home_team]['stl'][0][0] - 1)) + stl_h) / team_avgs[home_team]['stl'][0][0]

    # blocks
    blk_a = row['BLK_A']
    blk_h = row['BLK_H']
    team_avgs[away_team]['blk'][date] = team_avgs[away_team]['blk'][0][1]
    away_stats.append(team_avgs[away_team]['blk'][0][1])
    team_avgs[away_team]['blk'][0][0] += 1
    team_avgs[away_team]['blk'][0][1] = ((team_avgs[away_team]['blk'][0][1] * (team_avgs[away_team]['blk'][0][0] - 1)) + blk_a) / team_avgs[away_team]['blk'][0][0]
    team_avgs[home_team]['blk'][date] = team_avgs[home_team]['blk'][0][1]
    home_stats.append(team_avgs[home_team]['blk'][0][1])
    team_avgs[home_team]['blk'][0][0] += 1
    team_avgs[home_team]['blk'][0][1] = ((team_avgs[home_team]['blk'][0][1] * (team_avgs[home_team]['blk'][0][0] - 1)) + blk_h) / team_avgs[home_team]['blk'][0][0]

    # turnovers
    tov_a = row['TOV_A']
    tov_h = row['TOV_H']
    team_avgs[away_team]['tov'][date] = team_avgs[away_team]['tov'][0][1]
    away_stats.append(team_avgs[away_team]['tov'][0][1])
    team_avgs[away_team]['tov'][0][0] += 1
    team_avgs[away_team]['tov'][0][1] = ((team_avgs[away_team]['tov'][0][1] * (team_avgs[away_team]['tov'][0][0] - 1)) + tov_a) / team_avgs[away_team]['tov'][0][0]
    team_avgs[home_team]['tov'][date] = team_avgs[home_team]['tov'][0][1]
    home_stats.append(team_avgs[home_team]['tov'][0][1])
    team_avgs[home_team]['tov'][0][0] += 1
    team_avgs[home_team]['tov'][0][1] = ((team_avgs[home_team]['tov'][0][1] * (team_avgs[home_team]['tov'][0][0] - 1)) + tov_h) / team_avgs[home_team]['tov'][0][0]

    # winning and losing team
    # [away,home]
    winner[0] = row['Away_Win']
    winner[1] = row['Home_Win']


    # add to csv file
    row = []
    row = row + away_stats + home_stats + winner
    with open('./Processed_Data/preprocessed_games_18_19.csv', 'a') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(row)

with open('./current_averages_18_19.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    date_row = ['Date', '1/14/2019']
    header_row = ['Team','FG%','3P%','FT%','GP']
    wr.writerow(date_row)
    wr.writerow(header_row)
    for team,date in team_avgs.items():
        row = []
        row.append(team)
        row.append(team_avgs[team]['fg%'][0][1])
        row.append(team_avgs[team]['3p%'][0][1])
        row.append(team_avgs[team]['ft%'][0][1])
        row.append(team_avgs[team]['ft%'][0][0])
        wr.writerow(row)



# split up into training and testing
# 90% training, 10% testing
