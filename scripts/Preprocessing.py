# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 17:13:25 2025

@author: Administrator
"""
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os
import numpy as np
from sklearn.decomposition import PCA

league_ids = [700, 710, 720, 730, 740]

#standings = pd.read_excel(os.path.dirname(os.path.abspath(__file__)) + '\\base_data\\standings.xlsx')
standings = pd.read_csv('base_data/standings.csv')
standings = standings.sort_values(by='teamId')

# Filter standings to include only specified leagues
filtered_standings = standings[standings['leagueId'].isin(league_ids)]

# Get unique team IDs only from the filtered standings
team_ids = filtered_standings['teamId'].unique()

gamesPlayed = filtered_standings[['gamesPlayed']].to_numpy()
gamesPlayed = np.transpose(gamesPlayed)[0]
print(gamesPlayed)

# Create a dictionary mapping teamId to its rankings in each league
team_rank_matrix = np.zeros((5,len(team_ids)))
print(len(team_ids))

# Fill the matrix with team ranks
for i, league in enumerate(league_ids):
    league_data = standings[standings['leagueId'] == league]
    for j, team in enumerate(team_ids):
        team_rank = league_data[league_data['teamId'] == team]
        if len(team_rank) > 0:
            team_rank_matrix[i][j] = team_rank['teamRank']

# Load data
fixtures = pd.read_csv('base_data/fixtures.csv')[['leagueId', 'eventId']]
teamStats = pd.read_csv('base_data/teamStats.csv')

# Merge using an outer join to retain all eventIds
merged_df = pd.merge(teamStats,fixtures, on='eventId', how='outer')[['leagueId', 'eventId', 'teamId', 'possessionPct', 'accuratePasses', 'totalPasses', 'accurateCrosses', 'totalCrosses', 'totalLongBalls', 'accurateLongBalls', 'blockedShots', 'effectiveTackles', 'totalTackles', 'interceptions', 'effectiveClearance', 'totalClearance']]
merged_df = merged_df.dropna()
filtered_df = merged_df[merged_df['leagueId'].isin(league_ids)]

pivot = pd.pivot_table(data = filtered_df, values ={
    'possessionPct', 'accuratePasses', 'totalPasses', 'accurateCrosses', 'totalCrosses', 'totalLongBalls', 'accurateLongBalls', 'blockedShots', 'effectiveTackles', 'totalTackles', 'interceptions', 'effectiveClearance', 'totalClearance'}, index = {
'teamId',}, aggfunc = 'mean', dropna = True)

# Set display options to show more rows and columns
#pd.set_option('display.max_rows', None)  # Set to None to show all rows
#pd.set_option('display.max_columns', None)  # Set to None to show all columns
#pd.set_option('display.width', None)  # Prevent truncation of the table by width
#pd.set_option('display.max_colwidth', None)  # Prevent truncation of column values

# Display the pivot table
#print(pivot)
output_df = pd.DataFrame(pivot.to_records())
# Display the result
#print(team_ids)
#print(output_df['teamId'].to_numpy())

matrix = output_df.drop('teamId', axis=1)
X = matrix.to_numpy()
print(X)
#pca = PCA(n_components=10)
#pca.fit(X)

#print(pca.singular_values_)

k=3
kmeanModel = KMeans(k)
kmeanModel.fit(X)

# Add cluster labels to the original teamId column and convert to integers
output_df['ClusterLabel'] = kmeanModel.labels_.astype(int)

output_df['teamId'] = output_df['teamId'].astype(int)

# If you want to get them as an integer array:
team_cluster_array = output_df[['teamId', 'ClusterLabel']].to_numpy()
print(team_cluster_array)

