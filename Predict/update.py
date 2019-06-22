import pandas as pd
import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup as BS

# month dict
_MONTHS = {
    1: 'january',
    2: 'february',
    3: 'march',
    4: 'april',
    10: 'october',
    11: 'november',
    12: 'december'
}

# url to pull box score data from
bs_base = 'https://www.basketball-reference.com'
date_base = 'https://www.basketball-reference.com/boxscores/?'
today_base = 'https://www.basketball-reference.com/leagues/NBA_2019_games-{}.html'

def update_averages():
    print('Updating averages...')
    df = pd.read_csv('./current_averages_18_19.csv')
    start = pd.to_datetime(df['Date'].iloc[0])
    end = pd.to_datetime('today') - pd.Timedelta('1 day')
    date_range = pd.date_range(start, end, freq='1d')

    for d in date_range:
        # set up current day's url
        date = 'month='+str(d.month)+'&day='+str(d.day)+'&year='+str(d.year)
        date_url = date_base+date;
        #print("Games on: "+str(d.month)+"/"+str(d.day)+"/"+str(d.year))

        # request html from url using urllib2
        page = urlopen(date_url)

        # convert page to beautiful soup object
        soup = BS(page, 'html.parser')

        # get all links for each score
        links = soup.find_all('p',attrs={'class': 'links'})

        # first element in each links list is the box score
        for l in links:
            print("******GAME******")
            home_stats = []
            away_stats = []
            bs_l = str(l.contents[0])
            bs_full_url = bs_base+bs_l.split('"')[1]
            page = urlopen(bs_full_url)
            soup = BS(page, 'html.parser')

            # get names of home and away teams for keys in dictionary
            names = soup.find_all('a', attrs={'itemprop': 'name'})
            away_team = names[0].text
            home_team = names[1].text

            # first go through away team basic box score
            away = soup.find_all('table')[0]
            rows = away.find_all('tr')
            total_ind = len(rows)
            away_total = rows[total_ind-1]
            for stat in away_total:
                if stat.text != 'Team Totals' and stat.text != '':
                    away_stats.append(float(stat.text))
            print("Away Team: "+away_team)
            print(away_stats)

            # update dataframe of current averages with new one
            df = df.set_index('Team')
            df.loc[away_team, 'GP'] = df.loc[away_team, 'GP'] + 1
            df.loc[away_team, 'FG%'] = ((df.loc[away_team, 'FG%'] * (df.loc[away_team, 'GP'] - 1)) + away_stats[3]) / df.loc[away_team, 'GP']
            df.loc[away_team, '3P%'] = ((df.loc[away_team, '3P%'] * (df.loc[away_team, 'GP'] - 1)) + away_stats[6]) / df.loc[away_team, 'GP']
            df.loc[away_team, 'FT%'] = ((df.loc[away_team, 'FT%'] * (df.loc[away_team, 'GP'] - 1)) + away_stats[9]) / df.loc[away_team, 'GP']
            df = df.reset_index()

            # now do home team basic box score
            home = soup.find_all('table')[2]
            rows = home.find_all('tr')
            total_ind = len(rows)
            home_total = rows[total_ind - 1]
            for stat in home_total:
                if stat.text != 'Team Totals' and stat.text != '':
                    home_stats.append(float(stat.text))
            print("Home Team: "+home_team)
            print(home_stats)

            df = df.set_index('Team')
            df.loc[home_team, 'GP'] = df.loc[home_team, 'GP'] + 1
            df.loc[home_team, 'FG%'] = ((df.loc[home_team, 'FG%'] * (df.loc[home_team, 'GP'] - 1)) + home_stats[3]) / df.loc[home_team, 'GP']
            df.loc[home_team, '3P%'] = ((df.loc[home_team, '3P%'] * (df.loc[home_team, 'GP'] - 1)) + home_stats[6]) / df.loc[home_team, 'GP']
            df.loc[home_team, 'FT%'] = ((df.loc[home_team, 'FT%'] * (df.loc[home_team, 'GP'] - 1)) + home_stats[9]) / df.loc[home_team, 'GP']
            df = df.reset_index()

            print()

    print()
    print()
    df['Date'] = pd.to_datetime('today')
    df.to_csv('./current_averages_18_19.csv', index=False)
    print('Finished updating averages.')
    return df

def todays_games(avgDF):
    today = pd.to_datetime('today')
    date_range = pd.date_range(today, today, freq='1d')
    df = pd.DataFrame(columns=['Away','FG%_A','3P%_A','FT%_A','Home','FG%_H','3P%_H','FT%_H'])

    cur_url = today_base.format(_MONTHS[today.month])
    print("Games on: " + str(today.month) + "/" + str(today.day) + "/" + str(today.year))
    #print(cur_url)

    # request html from url using urllib2
    page = urlopen(cur_url)

    # convert page to beautiful soup object
    soup = BS(page, 'html.parser')

    # get all links for each score
    links = soup.find_all('p', attrs={'class': 'links'})
    games = soup.find_all('td', attrs={'data-stat': 'visitor_team_name'})

    # first element in each links list is the box score
    today = date_range.strftime('%Y%m%d')[0]
    for g in games:
        if today in g['csk']:
            print("******GAME******")
            tmpDF = pd.DataFrame(['Away', 'FG%_A', '3P%_A', 'FT%_A', 'Home', 'FG%_H', '3P%_H', 'FT%_H'])
            away_team = g.findChildren('a')[0].text
            home_team = g.parent.findChildren('td')[3].text
            print('Away Team: ' + away_team)
            print('Home Team: ' + home_team)

            # add to today's games dataframe
            awayDF = avgDF[avgDF['Team']==away_team][['Team','FG%','3P%','FT%']]
            awayDF = awayDF.rename(columns={'Team':'Away','FG%':'FG%_A','3P%':'3P%_A','FT%':'FT%_A'})
            homeDF = avgDF[avgDF['Team']==home_team][['Team','FG%','3P%','FT%']]
            homeDF = homeDF.rename(columns={'Team': 'Home', 'FG%': 'FG%_H', '3P%': '3P%_H', 'FT%': 'FT%_H'})
            #print(awayDF.head())
            #print(homeDF.head())
            tmpDF = pd.concat([awayDF.reset_index(drop=True),homeDF.reset_index(drop=True)], axis=1)
            #print(tmpDF.head())
            df = df.append(tmpDF, ignore_index=True)

            #print(df.head())
    return df
