
# Importing Dependency
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def adding_2022(d2):
    df = pd.read_csv('historical_RAPTOR_by_team.csv')
    # Using the player raptor at the end of a season for the next season
    df.season = df.season + 1
    # Removing 2022 season raptor to give current players
    e = df.loc[df['season'] == 2022]
    e.reset_index(inplace=True)
    e = e.drop(['index', 'player_id'], axis=1)

    d2 = d2.drop('SEASON', axis=1)
    #Updating the team abbreviations
    d2 = d2.replace({
        'BKN': 'BRK',
        'PHX': 'PHO'
    })
    d2.rename(columns={'PLAYER': 'player_name'
                       }, inplace=True)

    f = pd.merge(e, d2, on=['player_name'], how='outer')
    f['team'] = f['abbreviation'].copy()
    f = f.dropna(subset=['team'])
    f['season'] = f['season'].apply(lambda y: 2022)
    f['season'] = f['season'].apply(lambda z: 2022)
    #Dropping unnecessary columns
    arr = ['poss', 'mp',
           'raptor_offense', 'raptor_defense', 'raptor_total', 'war_total',
           'war_reg_season', 'war_playoffs', 'predator_offense',
           'predator_defense', 'predator_total', 'pace_impact']
    for x in arr:
        f[x] = f[x].fillna(f[x].mean())

    f['season_type'] = f['season_type'].fillna(f['season_type'].mode()[0])
    f = f.drop('abbreviation', axis=1)

    d1 = pd.read_csv('historical_RAPTOR_by_team.csv')
    d1 = d1.loc[:, ['season', 'player_name', 'team', 'season_type']]

    d3 = pd.merge(df, d1, on=['season', 'player_name', 'season_type'], how='inner')
    d3['team_x'] = d3['team_y'].copy()
    d3 = d3.drop('team_y', axis=1)
    d3.rename(columns={'team_x': 'team'}, inplace=True)
    d3 = d3.drop('player_id', axis=1)

    d = pd.concat([d3, f], ignore_index=True)
    return d


def random_2022(df, y):
    df = pd.read_csv('2021.csv')
    ran = df.sample(n=30)
    ran.reset_index(inplace=True)
    ran = ran.drop('index', axis=1)
    ran['abbreviation'] = ran['abbreviation'].apply(lambda x: y)
    df1 = pd.merge(df, ran, on=['SEASON', 'PLAYER'], how='outer')
    df1['abbreviation_y'] = df1['abbreviation_y'].fillna(df1['abbreviation_x'])
    df1.drop('abbreviation_x', axis=1, inplace=True)
    df1.rename(columns={'abbreviation_y': 'abbreviation'}, inplace=True)
    return df1, ran


def playoff_group(d):
    # Grouping the data
    m = d.groupby(['team', 'season'])

    # Summing up +/- raptor values to get team raptor
    n = m.sum()
    n.reset_index(inplace=True)

    # Changing outdated team abbreviations to recent abbrevations
    n = n.replace({
        'NOH': 'NOP',
        'CHA': 'CHO'
    })
    # Dropping columns of no use
    po_drop = ['war_reg_season', 'war_playoffs', 'poss', 'mp']
    p = n.drop(po_drop, axis=1)
    return p


def regular_season_group(d):
    # Grouping the data
    m = d.groupby(['season', 'team', 'season_type'])

    # Summing up +/- raptor values to get team raptor
    n = m.sum()
    n.reset_index(inplace=True)

    # Changing outdated team abbreviations to recent abbrevations
    n = n.replace({
        'NOH': 'NOP',
        'CHA': 'CHO'
    })
    # Grouping by season type to get Regular season Data
    o = n.groupby('season_type')
    p = o.get_group('RS')

    # Dropping columns of no use
    reg_season_drop = ['war_playoffs', 'war_total', 'season_type', 'poss', 'mp']
    p = p.drop(reg_season_drop, axis=1)
    return p


def random_elo(d2, y):
    e = d2.loc[d2['season'] == 2022]
    e.reset_index(inplace=True)
    e = e.drop(['index'], axis=1)
    n = e.sample()
    n.reset_index(inplace=True)
    n = n.drop('index', axis=1)
    m = n.team1[0]
    new_df = e.replace(m, y)
    d2.drop(d2.index[d2['season'] == 2022], inplace=True)
    d = pd.concat([d2, new_df], ignore_index=True)
    return d, m


def regular_season_games(df1):

    # Dropping columns of no use
    df1 = df1.drop(['date', 'neutral', 'carm-elo1_pre',
                    'carm-elo2_pre', 'carm-elo_prob1', 'carm-elo_prob2',
                    'carm-elo1_post', 'carm-elo2_post', 'raptor1_pre',
                    'raptor2_pre','raptor_prob1','raptor_prob2',
                    'elo1_post','elo2_post', 'importance','total_rating'], axis=1)

    # Dropping the playoff rows to process only regular season data
    play = ['t', 'q', 's', 'c', 'f', 'p']
    for p in play:
        df1.drop(df1.index[df1['playoff'] == p], inplace=True)

    # dropping the playoff column
    df1.drop('playoff', axis=1, inplace=True)

    # Dropping data from 1947 to 1978 since we have no raptor data earlier than 1978
    play = range(1946,1978)
    for p in play:
        df1.drop(df1.index[df1['season'] == p], inplace=True)

    # Resetting the indexing
    df1.reset_index(inplace=True)
    df1.drop('index', axis=1,inplace=True)

    # Creating a copy of the df1 for the purpose of determining the amount of wins and losses of every team
    df2 = df1.copy()

    # Creating a win Column to indicate wins or losses
    # 1 - Win
    # 0 - Loss
    df1['Win'] = df1.score1 - df1.score2
    df1['Win'] = df1['Win'].apply(lambda x : 0 if x<0 else 1)

    # Creating a Location Column to indicate Home and away games
    # 1 - Home
    # 0 - Away
    df1['Location'] = df1['team1'].copy()
    df1['Location'] = df1['Location'].apply(lambda x : 1 if x in x else 0)

    # Renaming columns to make Home team,Away teams and vice versa
    df2.rename(columns={'team1': 'team2',
                        'team2': 'team1',
                        'score1': 'score2',
                        'score2': 'score1',
                        'elo1_pre': 'elo2_pre',
                        'elo2_pre': 'elo1_pre',
                        'elo_prob1': 'elo_prob2',
                        'elo_prob2': 'elo_prob1'
                        }, inplace=True)

    # Creating a win Column to indicate wins or losses
    # 1 - Win
    # 0 - Loss
    df2['Win'] = df2.score1 - df2.score2
    df2['Win'] = df2['Win'].apply(lambda x: 0 if x < 0 else 1)

    # Creating a Location Column to indicate Home and away games
    # 1 - Home
    # 0 - Away
    df2['Location'] = df2['team1'].copy()
    df2['Location'] = df2['Location'].apply(lambda x: 0 if x in x else 1)

    # Concatting df1 and df2 to get a dataframe that has every teams win and loss
    df3 = pd.concat([df1, df2], ignore_index=True)
    return df3


def playoff_games(df1):
    # Dropping data from 1947 to 1977 since we have no raptor data earlier than 1978

    play = range(1946, 1978)
    for p in play:
        df1.drop(df1.index[df1['season'] == p], inplace=True)

    # Dropping columns of no use
    df1 = df1.drop(
        ['date', 'elo1_pre', 'elo_prob1', 'elo_prob2', 'elo2_pre', 'neutral', 'carm-elo1_pre', 'carm-elo2_pre',
         'carm-elo_prob1', 'carm-elo_prob2', 'carm-elo1_post', 'carm-elo2_post', 'raptor1_pre', 'raptor2_pre',
         'raptor_prob1', 'raptor_prob2', 'elo1_post', 'elo2_post', 'importance', 'total_rating'], axis=1)

    # dropping the playoff column
    df1 = df1.drop('playoff', axis=1)

    # Resetting the indexing
    df1.reset_index(inplace=True)
    df1.drop('index', axis=1, inplace=True)

    ##Creating a copy of the df1 for the purpose of determining the amount of wins and losses of every team
    df2 = df1.copy()

    # Creating a win Column to indicate wins or losses
    # 1 - Win
    # 0 - Loss
    df1['Win'] = df1.score1 - df1.score2
    df1['Win'] = df1['Win'].apply(lambda x: 0 if x < 0 else 1)

    # Creating a Location Column to indicate Home and away games
    # 1 - Home
    # 0 - Away
    df1['Location'] = df1['team1'].copy()
    df1['Location'] = df1['Location'].apply(lambda x: 1 if x in x else 0)

    # Renaming columns to make Home team,Away teams and vice versa
    df2.rename(columns={'team1': 'team2',
                        'team2': 'team1',
                        'score1': 'score2',
                        'score2': 'score1',
                        }, inplace=True)

    # Creating a win Column to indicate wins or losses
    # 1 - Win
    # 0 - Loss
    df2['Win'] = df2.score1 - df2.score2
    df2['Win'] = df2['Win'].apply(lambda x: 0 if x < 0 else 1)

    # Creating a Location Column to indicate Home and away games
    # 1 - Home
    # 0 - Away
    df2['Location'] = df2['team1'].copy()
    df2['Location'] = df2['Location'].apply(lambda x: 0 if x in x else 1)

    # Concatting df1 and df2 to get a dataframe that has every teams win and loss
    df3 = pd.concat([df1, df2], ignore_index=True)
    return df3


def games(p, df3):
    #Renaming team column to merge with dataset
    raptor = p.rename(columns={'team': 'team1'})

    #Creating second raptor for opposing team
    raptor2 = raptor.rename(columns={'team1': 'team2'})

    df4 = pd.merge(df3,raptor,on=['season','team1'], how='outer')
    df5 = pd.merge(df4,raptor2,on=['season','team2'], how='inner', suffixes=('_team1','_team2'))
    return df5


def all_rs(df5, conf):
    #Merging the dataset to the conference
    elo= pd.merge(df5,conf,on=['team1'], how='inner')

    #Creating a conference value
    # 1 - West
    # 0 - East
    elo['conf_value']=elo['Conf'].copy()
    elo['conf_value'] = elo['conf_value'].apply(lambda x : 1 if x == 'West' else 0)
    return elo


def all_po(df5, conf):
    df5 = df5.dropna()
    #Merging the dataset to the conference
    elo= pd.merge(df5,conf,on=['team1'], how='inner')

    #Creating a conference value
    # 1 - West
    # 0 - East
    elo['conf_value'] = elo['Conf'].copy()
    elo['conf_value'] = elo['conf_value'].apply(lambda x: 1 if x == 'West' else 0)
    return elo


def train_rs(df):
    d = df.dropna()
    X = d.drop(['Win','Conf','team1','team2','score1','score2'], axis=1)
    y = d['Win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    print ("Log Reg")
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)

    log_east_train_score = log.score(X_train, y_train)
    log_east_test_score = log.score(X_test, y_test)

    print(f"Training Data Score: {log_east_train_score}")
    print(f"Testing Data Score: {log_east_test_score}")
    return log


def test_rs(df, log):
    test = df.loc[df["season"] == 2022]
    tock = test.dropna()
    tack = test.loc[test['score1'].isnull() ]
    tack_test = tack.drop(['team1','team2','score1','score2','Win','Conf'], axis=1)
    predictions2022 = tack.loc[:,["team1","Conf"]]
    log_prediction = log.predict(tack_test).tolist()
    predictions2022["Win"] = log_prediction
    prevp= tock.loc[:,["team1","Conf","Win"]]
    df1= pd.concat([prevp,predictions2022], ignore_index=True)
    return df1


def season_result_east(df2):
    east = df2.get_group('East')
    east_conf = east.groupby('team1')
    conf = east_conf.sum()
    conf = conf.sort_values("Win", ascending=False)
    conf['Position'] = range(1, 16)
    conf.reset_index(inplace=True)
    conf.set_index('Position', inplace=True)
    return conf


def season_result_west(df2):
    west = df2.get_group('West')
    west_conf = west.groupby('team1')
    wconf = west_conf.sum()
    wconf = wconf.sort_values("Win", ascending=False)
    wconf['Position'] = range(1, 16)
    wconf.reset_index(inplace=True)
    wconf.set_index('Position', inplace=True)
    return wconf


def east_playoffs(conf):
    conf = conf.iloc[0:8]
    conf = conf.drop('Win', axis=1)
    return conf


def west_playoffs(wconf):
    wconf = wconf.iloc[0:8]
    wconf = wconf.drop('Win', axis=1)
    return wconf


def train_po(df1):
    X = df1.drop(['Win', 'Conf', 'team1', 'team2', 'score1', 'score2'], axis=1)
    y = df1['Win']
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    mod = XGBClassifier(use_label_encoder=False)
    eval_set = [(X_test1, y_test1)]
    mod.fit(X_train1, y_train1, early_stopping_rounds=10, eval_metric='logloss', eval_set=eval_set, verbose=True)

    log_east_train1 = mod.score(X_train1, y_train1)
    log_east_test1 = mod.score(X_test1, y_test1)

    print(f"Training Data Score: {log_east_train1}")
    print(f"Testing Data Score: {log_east_test1}")
    return mod


def playoff(home, away, year, conf, mod, post):
    PO = {
        'season': [year, year, year, year, year, year, year, year, year, year, year, year, year, year],
        'team1': [home, home, away, away, home, away, home, away, away, home, home, away, home, away],
        'team2': [away, away, home, home, away, home, away, home, home, away, away, home, away, home],
        'quality': [45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45],  ##Average Quality of all games
        'Location': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ##team1 plays at home
        'conf': [conf, conf, conf, conf, conf, conf, conf, conf, conf, conf, conf, conf, conf, conf]
    }
    df = pd.DataFrame(PO)

    # Renaming team column to merge with dataset
    raptor1 = post.rename(columns={'team': 'team1'})

    # Creating second raptor for opposing team
    raptor2 = raptor1.rename(columns={'team1': 'team2'})

    df1 = pd.merge(df, raptor1, on=['season', 'team1'], how='inner')
    df2 = pd.merge(df1, raptor2, on=['season', 'team2'], how='inner', suffixes=('_team1', '_team2'))
    df2['conf_value'] = df['conf'].copy()
    df2 = df2.drop('conf', axis=1)

    east_predictions_2019 = df2.loc[:, ["team1", "team2"]]
    df3 = df2.drop(['team1', 'team2'], axis=1)
    log_prediction = mod.predict(df3).tolist()

    east_predictions_2019["pred"] = log_prediction
    m = east_predictions_2019.groupby('team1')
    n = m.sum()
    n.reset_index(inplace=True)
    n = n.sort_values("pred", ascending=False)
    n.reset_index(inplace=True)
    o = n.team1[0]
    return o


def final(home, away, year, mod, post):
    PO = {
        'season': [year, year, year, year, year, year, year, year, year, year, year, year, year, year],
        'team1': [home, home, away, away, home, away, home, away, away, home, home, away, home, away],
        'team2': [away, away, home, home, away, home, away, home, home, away, away, home, away, home],
        'quality': [45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45],  # Average Quality of all games
        'Location': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # team1 plays at home
        'conf': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
    }
    df = pd.DataFrame(PO)

    # Renaming team column to merge with dataset
    raptor1 = post.rename(columns={'team': 'team1'})

    # Creating second raptor for opposing team
    raptor2 = raptor1.rename(columns={'team1': 'team2'})

    df1 = pd.merge(df, raptor1, on=['season', 'team1'], how='inner')
    df2 = pd.merge(df1, raptor2, on=['season', 'team2'], how='inner', suffixes=('_team1', '_team2'))
    df2['conf_value'] = df['conf'].copy()
    df2 = df2.drop('conf', axis=1)

    east_predictions_2019 = df2.loc[:, ["team1", "team2"]]
    df3 = df2.drop(['team1', 'team2'], axis=1)
    log_prediction = mod.predict(df3).tolist()

    east_predictions_2019["pred"] = log_prediction
    m = east_predictions_2019.groupby('team1')
    n = m.sum()
    n.reset_index(inplace=True)
    n = n.sort_values("pred", ascending=False)
    n.reset_index(inplace=True)
    o = n.team1[0]
    return o


def east_2nd(conf, mod, post):
    east_2nd_round = {
        'team1': [playoff(conf.team1[1], conf.team1[8], 2022, 0, mod, post),
                  playoff(conf.team1[3], conf.team1[6], 2022, 0, mod, post)],
        'team2': [playoff(conf.team1[4], conf.team1[5], 2022, 0, mod, post),
                  playoff(conf.team1[2], conf.team1[7], 2022, 0, mod, post)]
        }
    east_2nd = pd.DataFrame(east_2nd_round)
    return east_2nd


def west_2nd(wconf, mod, post):
    west_2nd_round = {
        'team1': [playoff(wconf.team1[1], wconf.team1[8], 2022, 1, mod, post),
                  playoff(wconf.team1[3], wconf.team1[6], 2022, 1, mod, post)],
        'team2': [playoff(wconf.team1[4], wconf.team1[5], 2022, 1, mod, post),
                  playoff(wconf.team1[2], wconf.team1[7], 2022, 1, mod, post)]
        }
    west_2nd = pd.DataFrame(west_2nd_round)
    return west_2nd


def east_final(east_2nd, mod, post):
    east_conff = {
    'team1': [playoff(east_2nd.team1[0], east_2nd.team2[0], 2022, 0, mod, post)],
    'team2': [playoff(east_2nd.team1[1], east_2nd.team2[1], 2022, 0, mod, post)]
    }
    east_finals = pd.DataFrame(east_conff)
    return east_finals


def west_final(west_2nd, mod, post):
    west_conff = {
        'team1': [playoff(west_2nd.team1[0], west_2nd.team2[0], 2022, 1, mod, post)],
        'team2': [playoff(west_2nd.team1[1], west_2nd.team2[1], 2022, 1, mod, post)]
        }
    west_finals = pd.DataFrame(west_conff)
    return west_finals


def predict():
    latest = pd.read_csv('2021.csv')
    with_2021 = adding_2022(latest)
    post = playoff_group(with_2021)
    reg = regular_season_group(with_2021)

    # Reading elo file
    elo = pd.read_csv('nba_elo.csv')
    rdf3 = regular_season_games(elo)
    pdf3 = playoff_games(elo)
    p_games = games(post, pdf3)
    r_games = games(reg, rdf3)

    # Reading the Conference dataset
    teams_conference = pd.read_csv('teams Conference.csv')
    post_season = all_po(p_games, teams_conference)
    regular_season = all_rs(r_games, teams_conference)

    # Training our Logistic Regression model
    log = train_rs(regular_season)
    test_reg = test_rs(regular_season, log)

    # Grouping test results by conference
    test_group = test_reg.groupby('Conf')
    east_result = season_result_east(test_group)
    west_result = season_result_west(test_group)

    east_conference = east_playoffs(east_result)
    west_conference = west_playoffs(west_result)

    mod = train_po(post_season)

    east_1st_round = [f'{east_conference.team1[1]} v {east_conference.team1[8]}, Winner = {playoff(east_conference.team1[1], east_conference.team1[8], 2022, 0, mod, post)} ',
                      f'{east_conference.team1[4]} v {east_conference.team1[5]}, Winner = {playoff(east_conference.team1[4], east_conference.team1[5], 2022, 0, mod, post)}',
                      f'{east_conference.team1[3]} v {east_conference.team1[6]}, Winner = {playoff(east_conference.team1[3], east_conference.team1[6], 2022, 0, mod, post)}',
                      f'{east_conference.team1[2]} v {east_conference.team1[7]}, Winner = {playoff(east_conference.team1[2], east_conference.team1[7], 2022, 0, mod, post)}']

    west_1st_round = [f'{west_conference.team1[1]} v {west_conference.team1[8]}, Winner = {playoff(west_conference.team1[1], west_conference.team1[8], 2022, 0, mod, post)}',
                      f'{west_conference.team1[4]} v {west_conference.team1[5]}, Winner = {playoff(west_conference.team1[4], west_conference.team1[5], 2022, 0, mod, post)}',
                      f'{west_conference.team1[3]} v {west_conference.team1[6]}, Winner = {playoff(west_conference.team1[3], west_conference.team1[6], 2022, 0, mod, post)}',
                      f'{west_conference.team1[2]} v {west_conference.team1[7]}, Winner = {playoff(west_conference.team1[2], west_conference.team1[7], 2022, 0, mod, post)}']

    east_2nd_round = east_2nd(east_conference, mod, post)
    west_2nd_round = west_2nd(west_conference, mod, post)

    eastern_2nd = [f'{east_2nd_round.team1[0]} v {east_2nd_round.team2[0]}, Winner = {playoff(east_2nd_round.team1[0], east_2nd_round.team2[0], 2022, 0, mod, post)}',
                   f'{east_2nd_round.team1[1]} v {east_2nd_round.team2[1]}, Winner = {playoff(east_2nd_round.team1[1], east_2nd_round.team2[1], 2022, 0, mod, post)}']

    western_2nd = [f'{west_2nd_round.team1[0]} v {west_2nd_round.team2[0]}, Winner = {playoff(west_2nd_round.team1[0], west_2nd_round.team2[0], 2022, 0, mod, post)}',
                   f'{west_2nd_round.team1[1]} v {west_2nd_round.team2[1]}, Winner = {playoff(west_2nd_round.team1[1], west_2nd_round.team2[1], 2022, 0, mod, post)}']

    east_finals = east_final(east_2nd_round, mod, post)
    west_finals = west_final(west_2nd_round, mod, post)

    conference_finals = [f'Eastern Conference : {east_finals.team1[0]} v {east_finals.team2[0]}',
                         f'Western Conference : {west_finals.team1[0]} v {west_finals.team2[0]}']

    east_conf_champion = playoff(east_finals.team1[0], east_finals.team2[0], 2022, 0, mod, post)
    eastch = f'The Eastern Conference Champion: {east_conf_champion}'

    west_conf_champion = playoff(west_finals.team1[0], west_finals.team2[0], 2022, 0, mod, post)
    westch = f'The Western Conference Champion: {west_conf_champion}'

    nba_champion = final(west_conf_champion, east_conf_champion, 2022, mod, post)
    return west_result, east_result, east_conference,west_conference, east_1st_round, west_1st_round, eastern_2nd, western_2nd, conference_finals, eastch, westch, nba_champion


def random_team(random):
    latest = pd.read_csv('2021.csv')
    (random_latest, random_team) = random_2022(latest, random)
    my_team = random_team.drop('SEASON', axis=1)
    with_2021 = adding_2022(random_latest)
    post = playoff_group(with_2021)
    reg = regular_season_group(with_2021)

    # Reading elo file
    elo = pd.read_csv('nba_elo.csv')
    (ran_elo, replaced) = random_elo(elo, random)
    rdf3 = regular_season_games(ran_elo)
    pdf3 = playoff_games(ran_elo)
    p_games = games(post, pdf3)
    r_games = games(reg, rdf3)

    # Reading the Conference dataset
    teams_conference = pd.read_csv('teams Conference.csv')
    teams_conference.replace(replaced, random, inplace=True)
    post_season = all_po(p_games, teams_conference)
    regular_season = all_rs(r_games, teams_conference)

    # Training our Logistic Regression model
    log = train_rs(regular_season)

    # Testing our model
    test_reg = test_rs(regular_season, log)

    # Grouping test results by conference
    test_group = test_reg.groupby('Conf')
    east_result = season_result_east(test_group)
    west_result = season_result_west(test_group)
    east_conference = east_playoffs(east_result)
    west_conference = west_playoffs(west_result)

    sss = east_conference.team1.values.tolist()
    ttt = west_conference.team1.values.tolist()

    mod = train_po(post_season)

    east_1st_round = [
        f'{east_conference.team1[1]} v {east_conference.team1[8]}, Winner = {playoff(east_conference.team1[1], east_conference.team1[8], 2022, 0, mod, post)} ',
        f'{east_conference.team1[4]} v {east_conference.team1[5]}, Winner = {playoff(east_conference.team1[4], east_conference.team1[5], 2022, 0, mod, post)}',
        f'{east_conference.team1[3]} v {east_conference.team1[6]}, Winner = {playoff(east_conference.team1[3], east_conference.team1[6], 2022, 0, mod, post)}',
        f'{east_conference.team1[2]} v {east_conference.team1[7]}, Winner = {playoff(east_conference.team1[2], east_conference.team1[7], 2022, 0, mod, post)}']

    west_1st_round = [
        f'{west_conference.team1[1]} v {west_conference.team1[8]}, Winner = {playoff(west_conference.team1[1], west_conference.team1[8], 2022, 0, mod, post)}',
        f'{west_conference.team1[4]} v {west_conference.team1[5]}, Winner = {playoff(west_conference.team1[4], west_conference.team1[5], 2022, 0, mod, post)}',
        f'{west_conference.team1[3]} v {west_conference.team1[6]}, Winner = {playoff(west_conference.team1[3], west_conference.team1[6], 2022, 0, mod, post)}',
        f'{west_conference.team1[2]} v {west_conference.team1[7]}, Winner = {playoff(west_conference.team1[2], west_conference.team1[7], 2022, 0, mod, post)}']

    if random in sss:
        east1st = east_1st_round
    else:
        east1st = None
    if random in ttt:
        west1st = west_1st_round
    else:
        west1st = None

    east_2nd_round = east_2nd(east_conference, mod, post)
    west_2nd_round = west_2nd(west_conference, mod, post)

    eastern_2nd = [
        f'{east_2nd_round.team1[0]} v {east_2nd_round.team2[0]}, Winner = {playoff(east_2nd_round.team1[0], east_2nd_round.team2[0], 2022, 0, mod, post)}',
        f'{east_2nd_round.team1[1]} v {east_2nd_round.team2[1]}, Winner = {playoff(east_2nd_round.team1[1], east_2nd_round.team2[1], 2022, 0, mod, post)}']

    western_2nd = [
        f'{west_2nd_round.team1[0]} v {west_2nd_round.team2[0]}, Winner = {playoff(west_2nd_round.team1[0], west_2nd_round.team2[0], 2022, 0, mod, post)}',
        f'{west_2nd_round.team1[1]} v {west_2nd_round.team2[1]}, Winner = {playoff(west_2nd_round.team1[1], west_2nd_round.team2[1], 2022, 0, mod, post)}']

    uuu = (east_2nd_round.team1.values.tolist() + east_2nd_round.team2.values.tolist())
    vvv = (west_2nd_round.team1.values.tolist() + west_2nd_round.team2.values.tolist())

    if random in uuu:
        east2nd = eastern_2nd
    else:
        east2nd = None
    if random in vvv:
        west2nd = western_2nd
    else:
        west2nd = None

    east_finals = east_final(east_2nd_round, mod, post)
    west_finals = west_final(west_2nd_round, mod, post)

    conference_finals = [
        f'Eastern Conference Finals {east_finals.team1[0]} v {east_finals.team2[0]}',
        f'Western Conference Finals {west_finals.team1[0]} v {west_finals.team2[0]}']

    aaa = (east_finals.team1.values.tolist() + east_finals.team2.values.tolist() + west_finals.team1.values.tolist() + west_finals.team2.values.tolist())

    if random in aaa:
        fconf = conference_finals
    else:
        fconf = None

    east_conf_champion = playoff(east_finals.team1[0], east_finals.team2[0], 2022, 0, mod, post)
    west_conf_champion = playoff(west_finals.team1[0], west_finals.team2[0], 2022, 0, mod, post)
    nba_champion = final(west_conf_champion, east_conf_champion, 2022, mod, post)

    if random in (east_conf_champion, west_conf_champion):
        ecc = east_conf_champion
        wcc = west_conf_champion
    else:
        ecc = None
        wcc = None

    if random in (east_conf_champion, west_conf_champion):
        nbam = f'{east_conf_champion} v {west_conf_champion}'
        nbach = nba_champion
    else:
        nbach = None
        nbam = None
    return my_team, west_result, east_result, east_conference, west_conference, \
           east1st, west1st, east2nd, west2nd, fconf, ecc, wcc, nbach, nbam