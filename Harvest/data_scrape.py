import os
from urllib.request import urlopen

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as BS

''' Season start and end inputs. Will harvest between these dates '''
# TODO: add dictionary for season start/end for each year
season_start = pd.to_datetime('10/16/2018')
season_end = pd.to_datetime('4/10/2019')

season = pd.date_range(start=season_start, end=season_end, freq='D')

# path to save raw data
RAW_PATH = '~/Documents/NBA_Game_Classifier/Data/Raw_Data'

# URLs for harvesting
BS_BASE = 'https://www.basketball-reference.com'
DATE_BASE = 'https://www.basketball-reference.com/boxscores/?'

away_columns = ['Away_Team',
                'MP_Away',
                'FG_Away',
                'FGA_Away',
                'FG%_Away',
                '3P_Away',
                '3PA_Away',
                '3P%_Away',
                'FT_Away',
                'FTA_Away',
                'FT%_Away',
                'ORB_Away',
                'DRB_Away',
                'TRB_Away',
                'AST_Away',
                'STL_Away',
                'BLK_Away',
                'TOV_Away',
                'PF_Away',
                'PTS_Away',
                ]

home_columns = ['Home_Team',
                'MP_Home',
                'FG_Home',
                'FGA_Home',
                'FG%_Home',
                '3P_Home',
                '3PA_Home',
                '3P%_Home',
                'FT_Home',
                'FTA_Home',
                'FT%_Home',
                'ORB_Home',
                'DRB_Home',
                'TRB_Home',
                'AST_Home',
                'STL_Home',
                'BLK_Home',
                'TOV_Home',
                'PF_Home',
                'PTS_Home',
                ]


def parse_box_score(table=None):
    '''
    Parses stats from box score html object and returns list

    :param table: table html object
    :return: stats as a list
    '''

    stats = []
    rows = table.find_all('tr')
    total_ind = len(rows)
    total = rows[total_ind - 1]
    for stat in total:
        if stat.text != 'Team Totals' and stat.text != '':
            stats.append(float(stat.text))

    return stats


def data_scrape_main():
    columns = ['Date'] + away_columns + home_columns

    statsDF = pd.DataFrame(columns=columns)

    # loop through all days and harvest
    for day in season:
        year = day.year
        month = day.month
        url = DATE_BASE + f'month={month}&day={day.day}&year={year}'

        print(f'Games on: {day}')
        print(f'Harvesting from: {url}')
        # request html from url and parse
        page = urlopen(url)
        soup = BS(page, 'html.parser')

        # get all links for each game of the day
        links = soup.find_all('p', attrs={'class': 'links'})

        # loop through game links and request again for box score
        for l in links:
            print('******GAME******')
            bs_l = str(l.contents[0])
            bs_full_url = BS_BASE + bs_l.split('"')[1]
            page = urlopen(bs_full_url)
            soup = BS(page, 'html.parser')

            # names of home and away teams
            names = soup.find_all('a', attrs={'itemprop': 'name'})
            away_team = names[0].text
            home_team = names[1].text

            away_table = soup.find_all('table')[0]
            home_table = soup.find_all('table')[2]

            away_stats = parse_box_score(table=away_table)
            home_stats = parse_box_score(table=home_table)

            print(f'Away Team: {away_team}')
            print(away_stats)

            print(f'Home Team: {home_team}')
            print(home_stats)

            print()

            # append game to stats dataframe
            game_stats = [day, away_team] + away_stats + [home_team] + home_stats

            gameDF = pd.DataFrame([game_stats], columns=columns)
            statsDF = statsDF.append(gameDF)

    statsDF['Winner'] = np.where(statsDF['PTS_Home'] > statsDF['PTS_Away'], 'H', 'A')
    statsDF.to_csv(os.path.join(RAW_PATH, f'games_{season_start.year}_{season_end.year}.csv'), index=False)

if __name__ == '__main__':
    data_scrape_main()
