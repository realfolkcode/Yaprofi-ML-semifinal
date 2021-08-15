import pandas as pd
import sys
import numpy as np
import networkx as nx
from catboost import CatBoostRegressor, CatBoost

def get_friend_school(G, df):
    mean_col = []
    std_col = []
    for u in df['uid'].values:
        arr = []
        if u in G.nodes:
            for v in G[u]:
                if 'school' in G.nodes[v]:
                    arr.append(G.nodes[v]['school'])
            mean_col.append(np.nanmean(arr))
            std_col.append(np.nanstd(arr))
        else:
            mean_col.append(np.nan)
            std_col.append(np.nan)
    df = df.assign(friend_school = mean_col)
    df = df.assign(friend_school_std = std_col)
    return df


def get_friend_reg(G, df):
    mean_col = []
    std_col = []
    for u in df['uid'].values:
        arr = []
        if u in G.nodes:
            for v in G[u]:
                if 'reg' in G.nodes[v]:
                    arr.append(G.nodes[v]['reg'])
            mean_col.append(np.nanmean(arr))
            std_col.append(np.nanstd(arr))
        else:
            mean_col.append(np.nan)
            std_col.append(np.nan)
    df = df.assign(friend_reg = mean_col)
    df = df.assign(friend_reg_std = std_col)
    return df


def main():
    # load data
    friends = pd.read_csv('tmp/data/friends.csv')
    X = pd.read_csv('tmp/data/test.csv')
    edu = pd.read_csv('tmp/data/testEducationFeatures.csv')
    groups = pd.read_csv('tmp/data/testGroups.csv')

    # prepare data
    G = nx.from_pandas_edgelist(friends, 'uid', 'fuid')
    X = pd.merge(X, edu, 'left')
    nx.set_node_attributes(G, X[['uid', 'school_education']].set_index('uid').to_dict()['school_education'], "school")
    nx.set_node_attributes(G, X[['uid', 'registered_year']].set_index('uid').to_dict()['registered_year'], "reg")
    # calculate degree
    degrees = pd.DataFrame(G.degree())
    degrees = degrees.rename(columns={0: 'uid', 1: 'degree'})
    X = pd.merge(X, degrees, 'left')
    # calculate average friends school education and registration year
    X = get_friend_school(G, X)
    X = get_friend_reg(G, X)
    # calculate the number of groups
    X = pd.merge(X, groups.groupby('uid', as_index=False).count().rename(columns={'gid': 'groups'}), 'left')
    X['groups'] = X['groups'].fillna(0)
    # calculate average group school education
    df_groups = pd.merge(X, groups, how='outer', on='uid')
    groups_school = df_groups.groupby('gid').mean()[['school_education', 'registered_year']]
    df_groups = pd.merge(df_groups, groups_school, how='left', on='gid')
    df_groups = df_groups.groupby('uid', as_index=False).agg({'school_education_y': 'mean', 'registered_year_y': 'mean'}).rename(columns={'school_education_y': 'group_school', 'registered_year_y': 'group_reg'})
    X = pd.merge(X, df_groups, 'left')

    y = pd.DataFrame(X['uid'])
    X = X.drop(columns=['uid'])

    # load model and predict
    model = CatBoost()
    model.load_model("model_group_reg_cv")
    best_iter = 1106
    model.shrink(best_iter)
    y = y.assign(age=model.predict(X))
    y.to_csv(sys.stdout, index=False)

if __name__ == '__main__':
    main()