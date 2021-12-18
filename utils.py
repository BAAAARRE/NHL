import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial


class Tools:
    @staticmethod
    @st.cache
    def prepare_data():
        df_player_info = pd.read_csv('data/player_info.csv')
        df_player_info['player_name'] = df_player_info['firstName'] + '_' + df_player_info['lastName']
        df_player_info['birth_year'] = df_player_info['birthDate'].str[:4].astype(int)
        df_player_info = df_player_info[['player_id', 'player_name']]

        df_game_skater_stats = pd.read_csv('data/game_skater_stats.csv')
        df_game_skater_stats['game_year'] = df_game_skater_stats['game_id'].astype(str).str[:4].astype(int)
        df_game_skater_stats = df_game_skater_stats.drop(['game_id', 'team_id'], 1)
        df_game_skater_stats_agg = df_game_skater_stats.groupby(['player_id', 'game_year'], as_index=False).sum()
        df_game_skater_stats_agg['timeOnIce'] = df_game_skater_stats_agg['timeOnIce'] / 3600

        final = df_game_skater_stats_agg.merge(df_player_info, how='left', on='player_id')
        final['uid'] = final['player_name'] + '_' + final['game_year'].astype(str) + '_' + final['player_id'].astype(
            str)
        return final

    @staticmethod
    def multi_filter(df, sel, var):
        if len(sel) == 0:
            df_sel = df
        elif len(sel) != 0:
            df_sel = df[df[var].isin(sel)]
        return df_sel

    @staticmethod
    def acp(df, X, y):
        X.append(y)
        df_acp = df[X].groupby(y).mean()

        n = df_acp.shape[0]
        p = df_acp.shape[1]
        sc = StandardScaler()
        Z = sc.fit_transform(df_acp)

        acp_ = PCA(svd_solver='full')
        coord = acp_.fit_transform(Z)

        eigval = (n - 1) / n * acp_.explained_variance_

        return df_acp, n, p, acp_, coord, eigval

    @staticmethod
    def get_indices_of_nearest_neighbours(df, Coords, n):
        indice = np.array(df.index)
        tree = spatial.cKDTree(Coords)
        res = tree.query(Coords, k=n)[1][:, 0:]
        res = pd.DataFrame(indice[res])
        return res

    @staticmethod
    def same_reco(df, ind):
        res = []
        for i in df.columns:
            res.append(df[df[0] == ind].iloc[:, i].to_string()[5:].lstrip())
        res = res[1:]
        for i in range(len(res)):
            st.write('Joueurs ressemblant n°' + str(i + 1) + ' : ' + res[i])
        return res

    @staticmethod
    def make_df_reco_final(df_select, sel_player, df_reco, X_sel):
        data_reco_sel = df_select[df_select['uid'] == sel_player]
        data_reco = df_select[df_select['uid'].isin(df_reco)].drop_duplicates(subset=['uid'])
        data_reco_sort = data_reco.set_index('uid').loc[df_reco].reset_index()
        df_reco_final = pd.concat([data_reco_sel, data_reco_sort]).reset_index().drop("index", axis=1)
        df_reco_final = df_reco_final[X_sel]
        return df_reco_final

    @staticmethod
    def var_pre_selection(pre_list, name, list_var, num_col):
        if num_col.checkbox(name, True):
            pre_list.extend(list_var)
        return pre_list
