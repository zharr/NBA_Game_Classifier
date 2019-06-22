import pandas as pd

df = pd.read_csv('all_picks.csv')
df = df[df['Act_Winner'] != -1]
pct = df[df['Pred_Winner'] == df['Act_Winner']].shape[0]/df.shape[0]
print(pct)