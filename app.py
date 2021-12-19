import streamlit as st
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
    df = tls.prepare_data()

    # Set Sidebar
    st.sidebar.title('Generals filters')
    slider_games = st.sidebar.slider('Number of games played', float(df['nb_match'].min()), float(df['nb_match'].max()),
                                     (float(df['nb_match'].min()), float(df['nb_match'].max())))
    # Configure generals filters
    df_nb_match = df[df['nb_match'].between(slider_games[0], slider_games[1])]
    general_select = df[df.isin(df_nb_match)].dropna()

    X_cols = ['nb_match', 'assists', 'goals', 'shots', 'hits', 'powerPlayGoals', 'powerPlayAssists',
              'penaltyMinutes', 'faceOffWins', 'faceoffTaken', 'takeaways', 'giveaways',
              'shortHandedGoals', 'shortHandedAssists', 'blocked', 'plusMinus', 'evenTimeOnIce',
              'shortHandedTimeOnIce', 'powerPlayTimeOnIce']

    y_col = 'uid'

    df_acp, n, p, acp_, coord, eigval = tls.acp(df=general_select, X=X_cols, y=y_col)

    if len(X_cols) > 0:
        if n >= p:
            st.title('Recommandateur')
            sel_simi = st.selectbox(' Joueur que tu aimes', sorted(df_acp.index))
            nb_simi = st.number_input("Nombre de joueurs les plus ressemblants", min_value=1, max_value=n - 1,
                                      value=3)
            df_near = tls.get_indices_of_nearest_neighbours(df_acp, coord, nb_simi + 1)
            recos = tls.same_reco(df_near, sel_simi)

            df_reco_final = tls.make_df_reco_final(general_select, sel_simi, recos, X_cols)
            st.write('\n')
            st.title("Données des joueurs recommandés")
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
            | Data source : [NHL](https://www.nhl.com/)""")


if __name__ == "__main__":
    main()
