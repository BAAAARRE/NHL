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
        """
        Import player_info, concat first and last name, keep only player id and name.
        Import game_skater_stats, create game year, group by player and game year, timeOnIce from second to 60 minutes
        Merge player_info to game_skater_stats, create an uid from player id, player name and game year
        :return: pd.DataFrame
        """
        df_player_info = pd.read_csv('data/player_info.csv')
        df_player_info['player_name'] = df_player_info['firstName'] + '_' + df_player_info['lastName']
        df_player_info = df_player_info[['player_id', 'player_name']]

        df_game_skater_stats = pd.read_csv('data/game_skater_stats.csv')
        df_game_skater_stats['game_year'] = df_game_skater_stats['game_id'].astype(str).str[:4].astype(int)
        df_game_skater_stats = df_game_skater_stats.drop(['game_id', 'team_id'], 1)
        df_game_skater_stats_agg = df_game_skater_stats.groupby(['player_id', 'game_year'], as_index=False).sum()
        df_game_skater_stats_agg['timeOnIce'] = df_game_skater_stats_agg['timeOnIce'] / 3600
        df_game_skater_stats_agg.rename(columns={'timeOnIce': 'nb_match'}, inplace=True)

        final = df_game_skater_stats_agg.merge(df_player_info, how='left', on='player_id')
        final['uid'] = final['player_name'] + '_' + final['game_year'].astype(str) + '_' + final['player_id'].astype(
            str)
        return final

    @staticmethod
    def acp(df, X, y):
        """
        Generic functions to make a PCA
        :param df: pd.DataFrame. Dataset for PCA
        :param X: list. Columns names for quantitative variables
        :param y: string. Column name of variable of individuals
        :return: df_acp, n, p, acp_, coord, eigval
        """
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
        """
        Calculate n nearest neighbours for each individuals with a K-D Tree
        :param df: pd.DataFrame. df_acp
        :param Coords: np.array. Coords od PCA
        :param n: int. Number of nearest neighbours to calculate
        :return: pd.DataFrame
        """
        indice = np.array(df.index)
        tree = spatial.cKDTree(Coords)
        res = tree.query(Coords, k=n)[1][:, 0:]
        res = pd.DataFrame(indice[res])
        return res

    @staticmethod
    def same_reco(df, ind):
        """
        Display the name of nearest neighbours of a selected individuals
        :param df: pd.DataFrame. Dataset from K-D Tree
        :param ind: string. Selected individual
        :return: list
        """
        res = []
        for i in df.columns:
            res.append(df[df[0] == ind].iloc[:, i].to_string()[5:].lstrip())
        res = res[1:]
        for i in range(len(res)):
            st.write('Joueur ressemblant n°' + str(i + 1) + ' : ' + res[i])
        return res

    @staticmethod
    def make_df_reco_final(df_select, sel_player, df_reco, X_sel):
        """
        Make dataframe reco
        :param df_select: pd.DataFrame. Initial dataset from prepare_data()
        :param sel_player: string. Selected player
        :param df_reco: pd.DataFrame. Dataset from same_reco()
        :param X_sel: list. Quantitatives columns to display
        :return: pd.DataFrame
        """
        data_reco_sel = df_select[df_select['uid'] == sel_player]
        data_reco = df_select[df_select['uid'].isin(df_reco)].drop_duplicates(subset=['uid'])
        data_reco_sort = data_reco.set_index('uid').loc[df_reco].reset_index()
        df_reco_final = pd.concat([data_reco_sel, data_reco_sort]).reset_index().drop("index", axis=1)
        df_reco_final = df_reco_final[X_sel]
        return df_reco_final
