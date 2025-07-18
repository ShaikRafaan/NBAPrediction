from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
import pandas as pd
import time

all_players = players.get_active_players()

player_data=[]

for player in all_players:
    player_id=player['id']
    name=player['full_name']

    try:
        info= commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df = info.get_data_frames()[0]
        player_data.append({
            'player_id': player_id,
            'player': name,
            'team': df.loc[0, 'TEAM_NAME'],
            'height': df.loc[0, 'HEIGHT'],
            'weight': df.loc[0, 'WEIGHT'],
            'position': df.loc[0, 'POSITION']
        })
        print(player_data)
        time.sleep(0.6)  # NBA API rate limiting safeguard
    except Exception as e:
        print(f"Failed for {name}: {e}")

players_df = pd.DataFrame(player_data)
print(players_df.head())


from nba_api.stats.endpoints import playercareerstats

season_stats = []

for player in all_players:
    player_id = player['id']
    name = player['full_name']

    try:
        stats = playercareerstats.PlayerCareerStats(player_id=player_id,timeout=60)
        stats_df = stats.get_data_frames()[0]

        # Filter for the latest season played
        latest_season = stats_df.iloc[-1]

        season_stats.append({
            'player_id': player_id,
            'player': name,
            'season': latest_season['SEASON_ID'],
            'team_id': latest_season['TEAM_ID'],
            'gp': latest_season['GP'],
            'pts': latest_season['PTS'],
            'ast': latest_season['AST'],
            'reb': latest_season['REB'],
            'fgm': latest_season['FGM'],
            'fga': latest_season['FGA'],
            'fg_pct': latest_season['FG_PCT'],
            'min': latest_season['MIN']
        })
        print(season_stats)

        time.sleep(1.5)

    except Exception as e: 
        print(f"Failed for {name}: {e}")

stats_df = pd.DataFrame(season_stats)
print(stats_df.head())


full_df = pd.merge(players_df, stats_df, on='player_id')
print(full_df.head())

full_df.to_csv("nba_players_full_stats.csv", index=False)

data=pd.read_csv("data/nba_players_full_stats.csv")
print("Identifying missing columns or inconsistent data: \n",data.isnull().sum())