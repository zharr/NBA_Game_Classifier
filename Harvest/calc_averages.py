import os

import pandas as pd

RAW_DATA = '~/Documents/NBA_Game_Classifier/Data/Raw_Data'
FEATURE_DATA = '~/Documents/NBA_Game_Classifier/Data/Feature_Data'

season = '2018_2019'

away_columns = ['MP_Away',
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

home_columns = ['MP_Home',
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


''' 
Record structure 

FG%_A,3P%_A,FT%_A,ORB_A,DRB_A,AST_A,STL_A,BLK_A,TOV_A,FG%_H,3P%_H,FT%_H,ORB_H,DRB_H,AST_H,STL_H,BLK_H,TOV_H,Winner

All columns average of all games before for each team.
Winner column: Home=1, Away=0

'''


def calc_averages_main():

    statsDF = pd.read_csv(os.path.join(RAW_DATA, f'games_{season}.csv'))
    statsDF['Date'] = pd.to_datetime(statsDF['Date'])
    statsDF = statsDF.sort_values(by=['Date'])

    # every row, take average of everything above (sorted by date) for each column
    # group by team, take rolling average of each statistic

    expanding_away = statsDF.groupby(['Away_Team'])[away_columns].apply(lambda x: x.expanding().mean().shift())
    expanding_home = statsDF.groupby(['Home_Team'])[home_columns].apply(lambda x: x.expanding().mean().shift())

    away_group = statsDF.groupby(['Away_Team'])[away_columns]
    home_group = statsDF.groupby(['Home_Team'])[home_columns]

    rolling_away = away_group.expanding().mean().shift().reset_index()
    rolling_away.to_csv(os.path.join(FEATURE_DATA, 'test.csv'), index=False)
    rolling_home = pd.DataFrame(home_group.expanding().mean().shift().reset_index().set_index('level_1'))

    statsDF = pd.merge(statsDF, expanding_away, right_index=True, left_index=True).dropna()
    statsDF = pd.merge(statsDF, expanding_home, right_index=True, left_index=True).dropna()

    statsDF = statsDF[[col for col in statsDF.columns if '_x' not in col]]
    statsDF.columns = [col.replace('_y', '') for col in statsDF.columns]

    statsDF.to_csv(os.path.join(FEATURE_DATA, f'featured_games_{season}.csv'), index=False)



if __name__ == '__main__':
    calc_averages_main()
