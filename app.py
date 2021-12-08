import streamlit as st
import pandas as pd

from utils import Tools as tls


def main():
    # Set configs
    st.set_page_config(
        layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
        page_title='NHL App',  # String or None. Strings get appended with "• Streamlit".
        page_icon=None,  # String, anything supported by st.image, or None.
    )

    # Load Data
    df = pd.read_csv('data.csv')

    # Set Sidebar
    st.sidebar.title('Navigation onglet')
    #     st.sidebar.title('Generals filters')
    #     sel_country = st.sidebar.multiselect('Select country', sorted(df['Nation'].unique()))
    #     sel_league = st.sidebar.multiselect('Select league', sorted(df['League'].unique()))
    #     sel_team = st.sidebar.multiselect('Select team', sorted(df['Squad'].unique()))
    #     sel_player = st.sidebar.multiselect('Select player', sorted(df['Player']))
    #     slider_games = st.sidebar.slider('Played Minutes', float(df['Minutes played divided by 90'].min()), float(df['Minutes played divided by 90'].max()), (float(df['Minutes played divided by 90'].min()), float(df['Minutes played divided by 90'].max())))
    #     st.sidebar.title('Graphics options')
    #     st.sidebar.write('\n')
    #     check_label = st.sidebar.checkbox('With labels')
    #     by_color = st.sidebar.selectbox('Color by', ['None', 'Nation', 'League', 'Squad'])
    #
    # # Configure generals filters
    #     df_country = multi_filter(df, sel_country, 'Nation')
    #     df_league = multi_filter(df, sel_league, 'League')
    #     df_team = multi_filter(df, sel_team, 'Squad')
    #     df_player = multi_filter(df, sel_player, 'Player')
    #
    #     df_games = df[df['Minutes played divided by 90'].between(slider_games[0],slider_games[1])]
    #
    #     general_select = df[df.isin(df_country) & df.isin(df_league) & df.isin(df_team) & df.isin(df_player) & df.isin(df_games)].dropna()
    #

    # Page

    X_cols = ['timeOnIce', 'assists', 'goals', 'shots', 'hits', 'powerPlayGoals', 'powerPlayAssists',
              'penaltyMinutes', 'faceOffWins', 'faceoffTaken', 'takeaways', 'giveaways',
              'shortHandedGoals', 'shortHandedAssists', 'blocked', 'plusMinus', 'evenTimeOnIce',
              'shortHandedTimeOnIce', 'powerPlayTimeOnIce', 'birth_year', 'height_cm', 'weight']

    y_cols = 'uid'

    df_acp, n, p, acp_, coord, eigval = tls.acp(df=df, X=X_cols, y=y_cols)

    if len(X_cols) > 0:
        if n >= p:
            st.title('Recommandateur')
            sel_simi = st.selectbox(' Joueur que tu aimes', sorted(df_acp.index))
            nb_simi = st.number_input("Nombre de joueurs les plus ressemblants", min_value=1, max_value=n - 1,
                                       value=3)
            df_near = tls.get_indices_of_nearest_neighbours(df_acp, coord, nb_simi + 1)
            recos = tls.same_reco(df_near, sel_simi)

            df_reco_final = tls.make_df_reco_final(df, sel_simi, recos, X_cols)
            st.write('\n')
            st.title("Données des joueurs recommandées")
            st.write('\n')
            st.dataframe(df_reco_final)

        else:
            st.error("Nombre de joueurs sélectionnées inférieur au nombre de variables explicatives.")
    else:
        st.error("Pas assez variables explicatives sélectionnées.")

    # Bottom page
    st.write("\n")
    st.write("\n")
    st.info("""By : Ligue des Datas [Instagram](https://www.instagram.com/ligueddatas/) 
    | Data source : [Sport Reference Data](https://www.sports-reference.com/)""")

if __name__ == "__main__":
    main()
