# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(12,5)})
# %load_ext autoreload
# %autoreload 2

# %%
from tqdm.notebook import tqdm

# %%
import requests

# %%
with open('token.txt', 'r') as f:
    API_TOKEN = f.readlines()[0][:-1]

# %%
import berserk

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)


# %%
# berserk.enums.PerfType.BLITZ

# %%
def color2bool(color):
    if color == 'white':
        return True
    else:
        return False

def bool2color(b):
    if b:
        return 'white'
    else:
        return 'black'

def flip_color(color):
    if color == 'white':
        return 'black'
    else:
        return 'white'


# %%
def my_color(playes_data):
    if playes_data['white']['user']['name'] == 'menisadi':
        return 'white'
    else:
        return 'black'


# %%
def opponent_name(playes_data):
    return playes_data[flip_color(my_color(playes_data))]['user']['name']


# %%
games = client.games.export_by_player(username='menisadi', perf_type='blitz', 
                                          rated=True, analysed=True, opening=True, 
                                          evals=True)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## One game

# %%
first_game = next(games)

# %%
playes = first_game['players']

meni_color = my_color(playes)

my_rating = first_game['players'][meni_color]['rating']
opponent_rating = first_game['players'][flip_color(meni_color)]['rating']
opponent = opponent_name(first_game['players'])

time = first_game['createdAt']

termination = first_game['status']

winner = first_game['winner']
did_i_win = winner == meni_color

openning = first_game['opening']['name']

moves = first_game['moves']
game_length = len(moves.split(' '))

my_accuracy = first_game['players'][meni_color]['analysis']['acpl']
opponent_accuracy = first_game['players'][flip_color(meni_color)]['analysis']['acpl']

# %%
new_row = pd.DataFrame([{
        'time': time,
        'opponent_name': opponent,
        'my_rating': my_rating,
        'my_color': meni_color,
        'result': winner,
        'did_I_win': did_i_win, 
        'termination_type': termination,
        'length_of_game': game_length,
        'my_average_centripawn_loss': my_accuracy,
        'opponent_average_centripawn_loss': opponent_accuracy
}])

# %% [markdown]
# ## And now we loop

# %%
df = pd.DataFrame(
    columns=['time', 'opponent_name', 'my_rating', 'my_color', 
             'result', 'termination_type', 'length_of_game', 
             'my_acpl', 'opponent_acpl', 
             'my_blunders', 'opponent_blunders',
             'opening'])

# %%
rows = []

for game in tqdm(games):
    playes = game['players']
    meni_color = my_color(playes)
    my_rating = game['players'][meni_color]['rating']
    opponent_rating = game['players'][flip_color(meni_color)]['rating']
    opponent = opponent_name(game['players'])
    time = game['createdAt']
    termination = game['status']
    try:
        winner = game['winner']
    except:
        winner = 'draw'  # Correcting the typo here from `windder` to `winner`
    did_i_win = bool(winner == meni_color)
    opening = game['opening']['name']
    moves = game['moves']
    game_length = len(moves.split(' '))
    my_accuracy = game['players'][meni_color]['analysis']['acpl']
    opponent_accuracy = game['players'][flip_color(meni_color)]['analysis']['acpl']
    my_blunders = game['players'][meni_color]['analysis']['blunder']
    opponent_blunders = game['players'][flip_color(meni_color)]['analysis']['blunder']

    rows.append({
        'time': time,
        'opponent_name': opponent,
        'my_rating': my_rating,
        'my_color': meni_color,
        'opening': opening,
        'result': winner,
        'did_I_win': did_i_win,
        'termination_type': termination,
        'length_of_game': game_length,
        'my_acpl': my_accuracy,
        'opponent_acpl': opponent_accuracy,
        'my_blunders': my_blunders,
        'opponent_blunders': opponent_blunders,
    })

df = pd.DataFrame(rows)

print('Done')


# %%
df = df.sort_values(by='time')
df = df.reset_index().drop('index', axis=1)

# %%
df['opening_core'] = df['opening'].apply(lambda o: o.split(':')[0])

# %%
openings_df = df.groupby(['opening_core', 'my_color'])['did_I_win'].agg(['mean', 'count']).sort_values(by='count', ascending=False)

# %%
openings_df[openings_df['count'] > 5].rename(columns={'mean': 'win_rate', 'count': 'games'})

# %%
df['total_acpl'] = df['my_acpl'] + df['opponent_acpl']

# %%
df['my_cpl'] = df['my_acpl'] * df['length_of_game']
df['opponent_cpl'] = df['opponent_acpl'] * df['length_of_game']
df['total_cpl'] = df['total_acpl'] * df['length_of_game']

# %%
df['total_blunders'] = df['my_blunders'] + df['opponent_blunders']

# %%
df[
['opponent_name', 'my_rating', 'total_blunders',
 'termination_type', 'did_I_win',  'length_of_game',
 'total_acpl', 'my_acpl', 'opponent_acpl']
].sort_values(by=[
    'total_acpl', 'total_blunders', 'my_acpl', 'opponent_acpl', 'length_of_game'
]).head(20)

# %%
selected_columns = ['opponent_name', 'my_rating', 'total_blunders',
                    'termination_type', 'did_I_win',  'length_of_game',
                    'total_acpl', 'my_acpl', 'opponent_acpl']
df[df['total_blunders'] ==0].sort_values(by='length_of_game', ascending=False).head()[selected_columns]

# %%
df.loc[df['length_of_game'] > 25, ['opponent_name', 'my_rating', 'total_blunders',
                                    'termination_type', 'did_I_win',  'length_of_game',
                                    'total_cpl', 'my_acpl', 'opponent_acpl'
                                   ]].sort_values(by=[
    'total_cpl', 'length_of_game'], ascending=[True, False]).head(10)

# %%
df = df.infer_objects()

# %%
# df.to_parquet('analyzed.parquet')

# %%
df.groupby('opening_core')['my_acpl'].agg(['min', 'max', 'count', 'mean', 'median']).sort_values(by='count', ascending=False).head()

# %%
df.groupby('opening_core')['length_of_game'].agg(['min', 'max', 'count', 'mean', 'median']).sort_values(by='count', ascending=False).head()

# %%
pivot_opennings = df.groupby(
    'opening_core'
)['termination_type'].agg(
    ['value_counts']
).pivot_table(index='opening_core', columns='termination_type', aggfunc='sum', fill_value=0)
pivot_opennings['sum'] = pivot_opennings.sum(axis=1)
pivot_opennings.sort_values(by='sum', ascending=False).head(8)

# %%
pivot_opennings.columns = ['draw', 'mate', 'outoftime',	'resign', 'stalemate', 'timeout', 'sum']


# %%
def normalize_row(row):
    row_sum = row.sum() - row['sum']  # Exclude the sum column
    if row_sum == 0:
        return row  # If sum is 0, return original row
    else:
        normalized_row = row.drop('sum').div(row_sum).round(3)
        normalized_row['sum'] = row['sum']
        return normalized_row


# %%
pivot_opennings_normalized = pivot_opennings.apply(normalize_row, axis=1).sort_values(by='sum', ascending=False)

# %%
pivot_opennings_normalized.head(8)

# %%
sns.heatmap(pivot_opennings_normalized.head(8).drop(['draw', 'stalemate', 'timeout', 'sum'], axis=1))

# %%
sns.scatterplot(data=df, x='my_acpl', y='opponent_acpl')

# %%
sns.scatterplot(data=df, x='length_of_game', y='total_acpl')

# %%
sns.scatterplot(data=df, x='my_rating', y='total_acpl')

# %%
sns.histplot(data=df, x='my_acpl', hue='did_I_win', bins=30, kde=True)
plt.title('My ACPL')

# %%
sns.histplot(data=df, x='opponent_acpl', hue='did_I_win', bins=30, kde=True)
plt.title('Opponent ACPL')

# %%
sns.histplot(data=df, x='total_acpl', hue='did_I_win', bins=30, kde=True)
plt.title('Total ACPL')

# %%

# %%
sns.boxplot(data=df[df['opening_core'].isin(df['opening_core'].value_counts().head().index)], 
            x='opening_core', y='my_acpl', hue='did_I_win')

# %%
sns.boxplot(data=df[df['opening_core'].isin(df['opening_core'].value_counts().head().index)], 
            x='opening_core', y='total_acpl', hue='did_I_win')

# %%
sns.boxplot(data=df[df['opening_core'].isin(df['opening_core'].value_counts().head().index)], 
            x='opening_core', y='my_acpl')

# %%
sns.countplot(data=df, x='total_blunders', hue='did_I_win')

# %%
sns.boxplot(data=df, x='did_I_win', y='total_blunders')

# %%
df.loc[df['total_blunders']==1, 'did_I_win'].value_counts()

# %%
df.loc[df['total_blunders']==1, ['did_I_win', 'termination_type']].value_counts()

# %% [markdown]
# ### Games per day

# %%
df["date"] = df["time"].dt.date

# %%
daily_stats = df.groupby('date').agg(
    games_played=('date', 'size'),
    games_won=('did_I_win', 'sum')
).reset_index()

# %%
daily_stats['win_ratio'] = daily_stats['games_won'] / daily_stats['games_played']

# %%
number_of_game_total = (daily_stats["games_played"] < 6).sum()

# %%
sns.lineplot(data=daily_stats.loc[daily_stats["games_played"] < 6], x="games_played", y="win_ratio")

# %%
sns.barplot(data=daily_stats.loc[daily_stats["games_played"] < 6], x="games_played", y="win_ratio")
plt.title(f"Comparing how much I play (per day) and how much I win (on that day)\n\nTotal number of games: {number_of_game_total}")

# %%
