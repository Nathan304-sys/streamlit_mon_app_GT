import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import joblib
import os
from PIL import Image
import base64
import io
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import qrcode
# try:
#     import openpyxl
#     _ = openpyxl  # Accès explicite au module pour le linter
# except ImportError:
#     st.error("Le package openpyxl n'est pas installé. Veuillez l'installer avec la commande: pip install openpyxl")
#     st.stop()
import streamlit as st




# # --- Authentification --- #
# VALID_USERNAME = os.environ.get("STREAMLIT_USERNAME", "nathan_exaucee")
# VALID_PASSWORD = os.environ.get("STREAMLIT_PASSWORD", "afriland2025") # You can change this password!

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# if not st.session_state.logged_in:
#     st.sidebar.empty() # Clear sidebar content for login page
#     st.title("Connexion à l'Application")
    
#     st.image("images/afriland_logo.png", width=150) 
   
#     st.markdown("<h3 style=\"text-align: center; color: #cc0000;\">Accès sécurisé aux données bancaires</h3>", unsafe_allow_html=True)
    
#     username = st.text_input("Nom d'utilisateur")
#     password = st.text_input("Mot de passe", type="password")

#     if st.button("Se connecter", help="Cliquez pour vous connecter à l'application"): 
#         if username == VALID_USERNAME and password == VALID_PASSWORD:
#             st.session_state.logged_in = True
#             st.success("Connexion réussie!")
#             st.rerun() # Rerun to display the app
#         else:
#             st.error("Nom d'utilisateur ou mot de passe incorrect.")
#     st.stop() # Stop execution until logged in
# # --- Fin Authentification --- #

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Défauts de Paiement - Afriland First Bank",
    #page_icon= st.image("images/afriland_logo.png", width=150),
    page_icon="💰",
    layout="wide"
)

# Style CSS personnalisé Afriland First Bank
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #1a1a1a;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #cc0000;
        color: white;
        padding: 0.8rem 1rem;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #990000;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .section-header {
        color: #cc0000;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #cc0000;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: transparent;
        color: white;
    }
    div[data-testid="stSidebar"] {
        background-color: #2a2a2a;
        padding: 1rem;
        color: white;
    }
    .stMarkdown {
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2a2a2a;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #2a2a2a;
    }
    </style>
    """, unsafe_allow_html=True)

################################################## En tete de l'application  ####################################


# BASE DE DONNEES
categorical_features= ['objet_credit_groupe', 'type', 'segment','profil_activite', 'secteur_risque', 'forme_juridique', 'reseau', 'cat_age_entreprise', 'statut']
# len(categorical_features)
numerical_features=['montant_credit', 'total_echeance',
       'capital_rembourse', 'nbre_ech',
       'taux_interet', 'nb_cpt', 'age_credit_jours', 'echa_impaye_avant',
       'cum_taux_paiement']
# leasing_filtered = leasing[numerical_features + categorical_features]
# len(numerical_features)
# leasing_filtered.head(2)

#Chargement de la base de données
#pd.read_csv(path,encoding='utf-8', sep=';')
@st.cache_data
def load_data():
    path = os.path.join("data", "leasing_filtered.csv")
    try:
        return pd.read_csv(path, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        # Try with different encoding if utf-8 fails
        return pd.read_csv(path, sep=';', encoding='latin-1')
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return None

base_leasing = load_data()
if base_leasing is None:
    st.error("Impossible de charger les données. Veuillez vérifier le fichier et son format.")
    st.stop()

if 'Unnamed: 0' in base_leasing.columns:
    base_leasing=base_leasing.drop('Unnamed: 0', axis=1)

col1, col2 = st.columns([1, 4])
with col1:
    st.image("images/afriland_logo.png", width=150)
with col2:
    st.title("Prédiction des Défauts de Paiement")
    st.markdown("---")
    

# Images for sections
def load_images():
    loaded_images = {}
    image_paths = {
        "data_presentation": "images/data_presentation.png",
        "data_analysis": "images/data_analysis.png",
        "modeling": "images/modeling.png",
        "customer_icon": "images/customer_icon.png",
        "feature_importance": "images/feature_importance.png",
        "coefficient_importance": "images/coefficient_importance.png",
        "nathan": "images/nathan.jpg",
        "exaucee": "images/exaucee.jpg"
    }
    for key, path in image_paths.items():
        try:
            loaded_images[key] = Image.open(path)
            print(f"Successfully loaded {path}")
        except FileNotFoundError:
            st.error(f"Erreur: L'image {path} n'a pas été trouvée. Veuillez vérifier le chemin et le nom du fichier.")
            print(f"FileNotFoundError: {path}")
            # For now, we'll add a placeholder if the file is not found to prevent further errors
            loaded_images[key] = None 
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'image {path}: {e}")
            print(f"Error loading {path}: {e}")
            loaded_images[key] = None
    return loaded_images

images = load_images()


###################################### Volet de Navigation de l'application ##############################################""

# Sidebar pour la navigation
st.sidebar.title("Navigation")
# page = st.sidebar.radio(
#     "Choisir une section",
#     ["Présentation des données", "Analyse exploratoire", "Modélisation", "Résultats"],
#     key="page_selection"
# )
# key="page_selection"
# if "page_selection" not in st.session_state:
#     st.session_state["page_selection"] = "Présentation des données"

page = st.sidebar.radio("Choisir une section", ["Présentation des données", "Analyse des données", "Modélisation", "À propos"], key="page_selection")

st.sidebar.markdown("---") # Add a separator
if st.sidebar.button("Se déconnecter", help="Cliquez pour vous déconnecter de l'application."):
    st.session_state.logged_in = False
    st.rerun()

#-------------------------------------Presentation des données -----------------------------------

if page == "Présentation des données":
    st.header("Présentation des Données")
    st.image(images["data_presentation"], caption="Aperçu de nos données", use_column_width=None)

    st.markdown("""
    <div class="section-header">Statistiques Clés du Jeu de Données</div>
    """, unsafe_allow_html=True)

    # Calcul des statistiques clés
    total_unique_clients = base_leasing['code_client'].nunique()
    total_unique_contracts = base_leasing['reference_lettrage'].nunique()

    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.metric(label="Total Clients Uniques", value=total_unique_clients,help="Données collectées (octobre 2022–mars 2025)")
    with col_stats2:
        st.metric(label="Total Contrats Uniques", value=total_unique_contracts, help="Données collectées (octobre 2022–mars 2025)")

    st.markdown("""
    <div class="section-header">Aperçu du Jeu de Données</div>
    """, unsafe_allow_html=True)
    st.dataframe(base_leasing.head())

    st.markdown("""
    <div class="section-header">Téléchargement du Jeu de Données</div>
    """, unsafe_allow_html=True)
    
    # Function to convert dataframe to excel for download
    def to_excel(df):
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
        processed_data = output.getvalue()
        return processed_data

    # Add a buffer for the excel file download
    # Ensure that `io` is imported at the top of the file along with other imports

    csv_file = base_leasing.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les données au format CSV",
        data=csv_file,
        file_name='leasing_filtered.csv',
        mime='text/csv',
        help="Téléchargez le jeu de données complet au format CSV"
    )

    excel_data = to_excel(base_leasing)
    st.download_button(
        label="Télécharger les données au format Excel",
        data=excel_data,
        file_name='leasing_filtered.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        help="Téléchargez le jeu de données complet au format Excel"
    )

#--------------------------------------------- Analyses des données ---------------------------------------------

elif page == "Analyse des données":
    st.header("Analyse Exploratoire des Données")
    st.image(images["data_analysis"], caption="Plongez dans les données", use_column_width=None)
    st.markdown("""
    <div class="section-header">Statistiques Descriptives des Échéances</div>
    """, unsafe_allow_html=True)

    #'reference_lettrage',
    st.dataframe(base_leasing[[ 'montant_credit', 'total_echeance',
       'capital_rembourse', 'nbre_ech',
       'taux_interet', 'nb_cpt', 'age_credit_jours', 'echa_impaye_avant',
       'cum_taux_paiement']].describe()) #base_leasing.describe() .groupby("reference_lettrage")

    st.markdown("""
    <div class="section-header">Visualisation des Variables Quantitatives/ Qualitative</div>
    """, unsafe_allow_html=True)

    # Identify quantitative columns (excluding 'statut' if it's numerical and not a feature)
    # Assuming 'statut' is the target variable and might be encoded as 0/1
    
    col1, col2= st.columns(2)

    with col1:
        # quantitative_cols = base_leasing.select_dtypes(include=np.number).columns.tolist()
        # if 'statut' in quantitative_cols:
        #     quantitative_cols.remove('statut')
        # if 'code_client' in quantitative_cols:
        #     quantitative_cols.remove('code_client')
        # if 'reference_lettrage' in quantitative_cols:
        #     quantitative_cols.remove('reference_lettrage')
        # if 'n_echance' in quantitative_cols:
        #     quantitative_cols.remove('n_echance')

        selected_quantitative_var = st.selectbox(
            "Sélectionnez une variable quantitative pour le Boxplot",
            numerical_features
        )
    
        if selected_quantitative_var:
            fig_boxplot = px.box(base_leasing, y=selected_quantitative_var, title=f"Boxplot de {selected_quantitative_var}",
                                color_discrete_sequence=["#cc0000"])
            st.plotly_chart(fig_boxplot, use_container_width=True)

            # st.markdown("""
            # <div class="section-header">Visualisation des Variables Qualitatives</div>
            # """, unsafe_allow_html=True)


        #     Categorical variables  : Index(['reference_lettrage', 'objet_credit_groupe', 'type', 'segment',
        #        'profil_activite', 'secteur_risque', 'forme_juridique', 'reseau',
        #        'cat_age_entreprise', 'Retard', 'statut'],
        #       dtype='object')
        # Numerical variables  : Index(['code_client', 'n_echance', 'montant_credit', 'total_echeance',
        #        'capital_rembourse', 'capital_restant', 'montant', 'nbre_ech',
        #        'taux_interet', 'nb_cpt', 'age_credit_jours', 'echa_impaye_avant',
        #        'cum_taux_paiement'],
        #       dtype='object')

    
    with col2:
        # qualitative_cols = base_leasing.select_dtypes(include='object').columns.tolist()
        # if 'statut' in qualitative_cols: # 'statut' might be object type if not encoded yet
        #     qualitative_cols.remove('statut')
        # if 'code_client' in qualitative_cols:
        #     qualitative_cols.remove('code_client')
        # if 'reference_lettrage' in qualitative_cols:
        #     qualitative_cols.remove('reference_lettrage')
        # if 'Retard' in qualitative_cols:
        #     qualitative_cols.remove('Retard')

        selected_qualitative_var = st.selectbox(
            "Sélectionnez une variable qualitative pour le Bar Chart",
            categorical_features
        )

        if selected_qualitative_var:
            # Calculate value counts and percentages
            value_counts = base_leasing[selected_qualitative_var].value_counts(normalize=True) * 100
            value_counts = value_counts.sort_values(ascending=False)
            
            fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Distribution de {selected_qualitative_var} (%)",
                           color_discrete_sequence=["#cc0000"])
            
            # Update layout to show percentages
            fig_bar.update_layout(
                yaxis_title="Pourcentage (%)",
                xaxis_title=selected_qualitative_var
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("""
    <div class="section-header">Description des Profils des Contrats Impayés</div>
    """, unsafe_allow_html=True)


    col3, col4= st.columns(2)
    with col3:
        st.subheader("Variables Quantitatives vs Statut")
        selected_quantitative_var_statut = st.selectbox(
            "Sélectionnez une variable quantitative pour la comparaison avec le statut",
            numerical_features, key="quant_vs_statut"
        )

        if selected_quantitative_var_statut:
            fig_boxplot_statut = px.box(base_leasing, x='statut', y=selected_quantitative_var_statut,
                                        color='statut', title=f"Boxplot de {selected_quantitative_var_statut} par Statut de Paiement",
                                        color_discrete_map={'Payé': '#000000', 'Impayé': '#cc0000'})
            st.plotly_chart(fig_boxplot_statut, use_container_width=True)

    with col4:
        st.subheader("Variables Qualitatives vs Statut")
        selected_qualitative_var_statut = st.selectbox(
            "Sélectionnez une variable qualitative pour la comparaison avec le statut",
            categorical_features, key="qual_vs_statut"
        )

        if selected_qualitative_var_statut:
            # # Calculate percentages for each category
            # value_counts = pd.crosstab(base_leasing[selected_qualitative_var_statut], 
            #                          base_leasing['statut'], 
            #                          normalize='index') * 100
            
            # # Create bar chart with improved readability
            # fig_hist_statut = px.bar(value_counts, 
            #                        title=f"Profil de {selected_qualitative_var_statut} par Statut de Paiement",
            #                        color_discrete_map={'Payé': '#000000', 'Impayé': '#cc0000'},
            #                        barmode='group',
            #                        text_auto='.1f')  # Automatically add percentage labels
            
            # # Update layout for better visualization
            # fig_hist_statut.update_layout(
            #     yaxis_title="Pourcentage (%)",
            #     xaxis_title=selected_qualitative_var_statut,
            #     showlegend=True,
            #     legend_title="Statut de Paiement",
            #     height=500,  # Increased height for better readability
            #     margin=dict(l=50, r=50, t=80, b=50),
            #     yaxis=dict(range=[0, 100])  # Force y-axis to go from 0 to 100%
            # )
            
            # # Customize text labels
            # fig_hist_statut.update_traces(
            #     texttemplate='%{y:.1f}%',
            #     textposition='outside',
            #     textfont_size=12,
            #     textfont_color='black'
                        # )
                        
                        # st.plotly_chart(fig_hist_statut, use_container_width=True)
             

            # Table de contingence normalisée par ligne (profil colonne)
            value_counts = pd.crosstab(
                base_leasing[selected_qualitative_var_statut],
                base_leasing['statut'],
                normalize='index'
            ) * 100

            # Remettre à plat pour que Plotly puisse lire (reset index + melt)
            value_counts_reset = value_counts.reset_index().melt(
                id_vars=selected_qualitative_var_statut,
                value_name='Pourcentage',
                var_name='Statut'
            )

            # Graphique à barres empilées 100%
            fig_hist_statut = px.bar(
                value_counts_reset,
                x=selected_qualitative_var_statut,
                y='Pourcentage',
                color='Statut',
                title=f"Profil de {selected_qualitative_var_statut} par Statut de Paiement",
                text_auto='.1f',
                color_discrete_map={'Payé': '#000000', 'Impayé': '#cc0000'},
            )

            # Personnalisation
            fig_hist_statut.update_layout(
                barmode='stack',
                yaxis_title="Pourcentage (%)",
                xaxis_title=selected_qualitative_var_statut,
                showlegend=True,
                legend_title="Statut de Paiement",
                height=500,
                margin=dict(l=50, r=50, t=80, b=50),
                yaxis=dict(range=[0, 100]),
                xaxis_tickangle=45  # Inclinaison des labels en abscisse
            )

            # Texte sur les barres
            fig_hist_statut.update_traces(
                texttemplate='%{y:.1f}%',
                textposition='inside',
                textfont_size=12,
                textfont_color='white'
            )

            # Affichage dans Streamlit
            st.plotly_chart(fig_hist_statut, use_container_width=True)

    st.markdown("""
    <div class="section-header">Corrélation des Variables</div>
    """, unsafe_allow_html=True)

    # For correlation matrix, we need only numerical columns
    numeric_df_corr = base_leasing[numerical_features].corr()
    fig_corr = px.imshow(numeric_df_corr, text_auto=True, aspect="auto",
                         title="Matrice de Corrélation des Variables Quantitatives",
                         color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    <div class="section-header">Tests Statistiques (Khi-deux)</div>
    """, unsafe_allow_html=True)
    
    st.info("Pour effectuer le test du Khi-deux, assurez-vous que la colonne 'statut' est correctement encodée (par exemple, 0 pour Payé, 1 pour Impayé).")

    col_chi2_1, col_chi2_2 = st.columns(2)
    with col_chi2_1:
        chi2_var1 = st.selectbox("Sélectionnez la première variable qualitative",categorical_features , key="chi2_var1") #qualitative_cols.append("statut")
    with col_chi2_2:
        chi2_var2 = st.selectbox("Sélectionnez la deuxième variable qualitative",categorical_features, key="chi2_var2")

    if st.button("Exécuter le test du Khi-deux"):
        if chi2_var1 and chi2_var2 and chi2_var1 != chi2_var2:
            # Ensure 'statut' is part of the data being analyzed, or that the selected qualitative columns are correct.
            # For now, let's assume 'statut' is relevant for the interpretation of chi-2 test.
            # We'll use the selected qualitative variables for the test itself.
            
            # Create a contingency table
            contingency_table = pd.crosstab(base_leasing[chi2_var1], base_leasing[chi2_var2])

            # Perform the Chi-squared test
            chi2, p_val, dof, expected = chi2_contingency(contingency_table)

            st.subheader(f"Résultats du test du Khi-deux entre {chi2_var1} et {chi2_var2}")
            st.write(f"Statistique du Khi-deux: {chi2:.2f}")
            st.write(f"Valeur p: {p_val:.3f}")
            st.write(f"Degrés de liberté: {dof}")

            st.subheader("Interprétation")
            if p_val < 0.05:
                st.success("Il existe une relation significative entre les deux variables qualitatives sélectionnées (p < 0.05).")
            else:
                st.warning("Il n'y a pas de relation significative entre les deux variables qualitatives sélectionnées (p >= 0.05).")
        else:
            st.warning("Veuillez sélectionner deux variables qualitatives différentes pour le test du Khi-deux.")

#--------------------------------------------- Modélisation ----------------------------------------- 

elif page == "Modélisation":
    st.header("Modélisation et Prédiction Avancée")
    st.image(images["modeling"], caption="Prédire l'avenir", use_column_width=None)

    # Importance des Caractéristiques du Modèle (always visible)
    st.markdown("""
    <div class="section-header">Importance des Caractéristiques du Modèle de Regression Logistique </div>
    """, unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.image(images["feature_importance"], caption="Aperçu des variables influentes")
    with col6:
        st.image(images["coefficient_importance"], caption="Aperçu des coefficients des variables influencant la probabilité de défaut", use_column_width=None)

    # Radio selector for prediction mode
    prediction_mode = st.radio(
        "Choisissez votre mode de prédiction",
        ("Prédiction par Contrat Existant", "Prédiction par Simulation de Données"),
        key="prediction_mode_selection"
    )

    if prediction_mode == "Prédiction par Contrat Existant":
        st.markdown("""
        <div class="section-header">Prédiction du Risque de Défaut par Contrat</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        Sélectionnez un contrat pour prédire son risque de défaut le mois prochain. La prédiction sera basée sur les caractéristiques de sa dernière échéance.
        """, unsafe_allow_html=True)

        unique_contracts = base_leasing[['code_client', 'reference_lettrage']].drop_duplicates()

        selected_client_ref = st.selectbox(
            "Sélectionnez un Client et Référence de Contrat",
            options=unique_contracts.apply(lambda row: f"{row['code_client']} - {row['reference_lettrage']}", axis=1).tolist()
        )

        if selected_client_ref:
            client_code = selected_client_ref.split(" - ")[0]
            ref_lettrage = selected_client_ref.split(" - ")[1]

            contract_data = base_leasing[(base_leasing['code_client'] == client_code) |
                                         (base_leasing['reference_lettrage'] == ref_lettrage)]
            
            latest_date = contract_data['date_echeance'].max()
            
            latest_installments = contract_data[contract_data['date_echeance'] == latest_date]

            if not latest_installments.empty:
                st.write(f"Caractéristiques de la dernière échéance pour le contrat {ref_lettrage} du client {client_code}:")
                df_for_prediction = latest_installments[numerical_features + categorical_features]
                st.dataframe(df_for_prediction)
            else:
                st.warning(f"La dernière écheance du contrat {ref_lettrage} du client {client_code} n'existe pas")

            if st.button("Prédire le risque de défaut"):
                st.markdown("---")
                st.markdown("### Résultat de la Prédiction")

                def model_final(model, X, threshold=0):
                    if not isinstance(X, pd.DataFrame):
                        X = pd.DataFrame(X)
                    decision_function_result = model.decision_function(X)
                    predict_proba_result = model.predict_proba(X)[:,1]
                    return decision_function_result > threshold, predict_proba_result
                
                @st.cache_resource
                def load_model(path):
                    return joblib.load(path)
                
                filename_joblib = 'modeles/lOGISTIC_REGRESSION_MODEL_3.joblib'
                loaded_model_joblib = load_model(filename_joblib)
                y_pred_loaded = model_final(loaded_model_joblib, df_for_prediction)
                
                prediction = y_pred_loaded[0]
                probabilite = y_pred_loaded[1][0]

                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = float(probabilite) * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ffcccc"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    },
                    title = {'text': "Risque de Défaut"}
                ))

                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)

                if float(probabilite) < 0.3:
                    st.success(" Faible risque de défaut")
                    st.write("Le contrat leasing souscrit par le client à moins de risque de tombée en impayé à la prochaine échéance")

                elif float(probabilite) < 0.7:
                    st.warning(" Risque modéré de défaut")
                    st.write("Le contrat leasing souscrit par le client à un risque modérée de tombée en impayé à la prochaine échéance")
                else:
                    st.error("Risque élevé de défaut")
                    st.write("Le contrat leasing souscrit par le client à un risque très élevé de tombée en impayé à la prochaine échéance")

    elif prediction_mode == "Prédiction par Simulation de Données":
        st.markdown("""
        <div class="section-header">Prédiction du Risque de Défaut par Simulation de Données</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        Entrez manuellement les caractéristiques du client et du contrat pour simuler une prédiction de risque de défaut.
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Caractéristiques du Client</div>', unsafe_allow_html=True)
            st.image(images["customer_icon"], width=100)
            
            segment = st.selectbox(
                "Segment de l'entreprise",
                ['GE', 'ME', 'PE', 'INS', 'TPE', 'Other', 'PAR'], 
                help="GE: Grande Entreprise, ME: Moyenne Entreprise, PE: Petite Entreprise, etc."
            )
            type_entreprise = st.selectbox(
                "Type d'entreprise",
                ['Société', 'Entreprise Individuelle']
            )
            
            forme_juridique = st.selectbox(
                "Forme juridique",
                ['SA','SARL', 'EURL', 'Other'] 
            )
            
            profil_activite = st.selectbox(
                "Profil d'activité",
                ['Autres', "Organisme de l'Etat" , 'Entreprise Privee Individuelle',
                 'Other']
                 
            )
            
            secteur_risque = st.selectbox(
                "Secteur d'activité",
                ['Commerce', 'Secteur Public', 'Activités Agro-Pastorales',
                  'Industries', 'Construction',
                 'Energie', 'Production Autres Services', 'Transports'])
                 
            
            reseau = st.selectbox(
                "Réseau d'affectation",
                ['Yaounde Centre', 'Douala Centre', 'Ouest', 'Zone Nord', 'Douala Nord',
                 'Douala Sud', 'Yaounde Sud', 'Est', 'Yaounde Nord']
            )
            
            cat_age_entreprise = st.selectbox(
                "Âge de l'entreprise",
                ['20+ ans', '10-20 ans', '3-10 ans', '0-3 ans']
            )
        
        with col2:
            st.markdown('<div class="section-header">Caractéristiques du Contrat</div>', unsafe_allow_html=True)
            st.image("https://img.icons8.com/color/96/000000/contract.png", width=100)
            
            montant_credit = st.number_input("Montant du crédit", min_value=0, value=10000)
            total_echeance = st.number_input("Montant de l'échéance", min_value=0, value=1000)
            capital_rembourse = st.number_input("Capital remboursé", min_value=0, value=0)
            nbre_ech = st.number_input("Nombre total d'échéances", min_value=1, value=12)
            taux_interet = st.slider("Taux d'intérêt (%)", 0.0, 30.0, 5.0)
            age_credit_jours = st.number_input("Âge du crédit (jours)", min_value=0, value=0)
            
            objet_credit = st.selectbox(
                "Objet du crédit",
                ['VOITURE', 'AUTRE', 'CAMION', 'MACHINE', 'BUS', 'REMORQUE', 'ENGIN']
            )
        
        st.markdown('<div class="section-header">Comportement de Paiement</div>', unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/000000/payment-history.png", width=100)
        
        col3, col4 = st.columns(2)
        
        with col3:
            cum_taux_paiement = st.slider("Cumul du Taux de paiement (%)", 0.0, 100.0, 10.0)
        
        with col4:
            nb_cpt = st.number_input("Nombre de comptes", min_value=1, value=1)

        if st.button("Prédire le risque de défaut (Simulation)"):
            st.markdown("---")
            st.markdown("### Résultat de la Prédiction (Simulation)")
            
            df_prediction_simulated = pd.DataFrame({
                    'total_echeance':total_echeance, 'capital_rembourse':capital_rembourse, 'nbre_ech':nbre_ech,
                    'taux_interet':taux_interet, 'age_credit_jours':age_credit_jours, 'nb_cpt':nb_cpt, 'cum_taux_paiement':cum_taux_paiement,
                    'objet_credit_groupe':objet_credit, 'type':type_entreprise, 'segment':segment, 'profil_activite':profil_activite,
                    'secteur_risque':secteur_risque, 'forme_juridique':forme_juridique, 'reseau':reseau,
                    'cat_age_entreprise':cat_age_entreprise},index=[1])

            def model_final(model, X, threshold=0):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                decision_function_result = model.decision_function(X)
                predict_proba_result = model.predict_proba(X)[:,1]
                return decision_function_result > threshold, predict_proba_result
            
            @st.cache_resource
            def load_model(path):
                return joblib.load(path)
            
            filename_joblib = 'modeles/lOGISTIC_REGRESSION_MODEL_3.joblib'
            loaded_model_joblib = load_model(filename_joblib)
            y_pred_loaded_simulated = model_final(loaded_model_joblib, df_prediction_simulated)
            
            prediction_simulated = y_pred_loaded_simulated[0]
            probabilite_simulated = y_pred_loaded_simulated[1][0]

            fig_simulated = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = float(probabilite_simulated) * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ffcccc"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                },
                title = {'text': "Risque de Défaut (Simulation)"}
            ))

            fig_simulated.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig_simulated, use_container_width=True)

            if float(probabilite_simulated) < 0.3:
                st.success(" Faible risque de défaut (Simulation)")
                st.write("Le contrat leasing simulé à moins de risque de tombée en impayé à la prochaine échéance")

            elif float(probabilite_simulated) < 0.7:
                st.warning(" Risque modéré de défaut (Simulation)")
                st.write("Le contrat leasing simulé à un risque modérée de tombée en impayé à la prochaine échéance")
            else:
                st.error("Risque élevé de défaut (Simulation)")
                st.write("Le contrat leasing simulé à un risque très élevé de tombée en impayé à la prochaine échéance")

#----------------------------------------- Avant Propos -------------------------------------------------

else:  # À propos
    st.header("Rapport d'étude")
    st.markdown("""
    <div class="section-header">Résumé de l'étude</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    **
Cette étude vise à identifier les facteurs déterminants des impayés dans les financements leasing à Afriland
First Bank et à développer un modèle prédictif du risque de défaut au niveau de chaque échéance. Pour ce
faire, nous avons exploité une base de données consolidée regroupant 15 096 échéances de crédit-bail
(octobre 2022–mars 2025) issues de plusieurs sources internes à la banque (données client,
caractéristiques des contrats et transactions bancaires).

Les techniques employées comprennent des
analyses univariées et bivariées, des tests d'indépendances pour évaluer les liens entre
variables explicatives et statut de paiement, ainsi qu'une modélisation par apprentissage automatique
(régression logistique, Random Forest, XGBoost, LightGBM) pour prédire la probabilité de défaut à la
prochaine échéance 

Nous avons testé quatre algorithmes (régression logistique, Random Forest, XGBoost, LightGBM) en
traitant le déséquilibre (46 % impayés vs 54 % payés) via une partition train/test (80 %/20 %) et
normalisation. La régression logistique s'est révélée la plus stable (AUC ≈ 0,93, accuracy ≈ 90 %) sans
surapprentissage, tandis que les modèles d'ensemble, bien qu'obtenant 100 % de précision en
apprentissage, souffraient d'un surapprentissage marqué. Les facteurs les plus influents dans la régression
logistique sont, dans l'ordre, le cumul du taux de paiement, le nombre d'échéance, l'age du crédit puis le montant
** 
    """)

    st.markdown("""
    <div class="section-header">Télécharger le Rapport Complet</div>
    """, unsafe_allow_html=True)

    try:
        with open("docs/rapport_etude.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()
            st.download_button(
                label="Télécharger le rapport PDF",
                data=PDFbyte,
                file_name="rapport_etude.pdf",
                mime="application/pdf",
                help="Cliquez pour télécharger le rapport complet de l'étude au format PDF."
            )
    except FileNotFoundError:
        st.error("Le fichier 'rapport_etude.pdf' n'a pas été trouvé dans le dossier 'docs'. Veuillez le placer là.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement du rapport PDF : {e}")

    st.markdown("---")
    st.header("À propos de l'application ")
    st.markdown("""
    Cette application permet de prédire le risque de défaut de paiement en utilisant des modèles de machine learning.
    
    ### Fonctionnalités
    - Prédiction du risque de défaut
    - Visualisation des données
    - Interface utilisateur intuitive
    
    ### Variables utilisées
    1. Caractéristiques du Client: 
       Segment de l'entreprise, Type d'entreprise,Forme juridique, Profil d'activité, Secteur d'activité, Réseau d'affectation, Âge de l'entreprise,
    
    2. Caractéristiques du Contrat: 
        Montant du crédit,  Échéances, Capital remboursé/restant, Taux d'intérêt, Objet du crédit
    
    3. Comportement de Paiement: 
       Cumul du Taux de paiement, Nombre de comptes
    """)

    # Section pour le QR Code
    st.markdown("""
    <div class="section-header">Accéder à l'Application par QR Code</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    Scannez ce code QR avec votre appareil mobile pour accéder directement à l'application.
    """)

    # Remplacez cette URL par l'URL de déploiement de votre application si elle n'est pas locale
    app_url = "http://localhost:8501"

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(app_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convertir l'image PIL en bytes pour Streamlit
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.image(byte_im, caption=f"QR Code pour l'URL: {app_url}", width=200)
        # use_container_width 



    
    st.header("À propos de nous")
    st.write("""
    Nous sommes deux élèves ingénieurs statisticiens économistes en formation, comme voie d'approfondissment Data Science et Marketing à l'ISSEA.
    Ayant développé cette application.
             
    """)
    st.markdown('---')
    
    # dylan = Image.open(main_dir("dylan.jpg"))
    # dylan = dylan.resize((200,300))
    st.markdown(
        f"""
        ##### NSIMOUESSA Dieuveil Nathan
        """,
        unsafe_allow_html=True,
    )
    col_im,col_ad=st.columns(2)
    with col_im:
        st.image(images["nathan"],use_column_width=None)

    with col_ad:
        st.markdown(
        f"""
        - 📧 Email:  <student.nsimouessa.dieuveil@issea-cemac.org> ou <nsimouessa@gmail.com>
        - Linkedin:  https://www.linkedin.com/ Nathan Nsimouessa
        - Contact: +237 677128351 ou +242 069168487
        """,
        unsafe_allow_html=True,
    )
    st.markdown('---')

    # mpolah = Image.open(main_dir("mpolah.jpg"))
    # mpolah = mpolah.resize((200,300))
    st.markdown(
        f"""
        ##### MISSENGUE Moloumbou Exaucée
        """,
        unsafe_allow_html=True,
    )
    col_im,col_ad=st.columns(2)
    with col_im:
        st.image(images["exaucee"], use_column_width=None)
    with col_ad:
        st.markdown(
        f"""
        - 📧 Email:  <emissenguemoulombo@gmail.com>
        - Linkedin:  http://www.linkedin.com/Missengue Exaucée
        
        """,
        unsafe_allow_html=True,
    )
    

    

# Footer
st.markdown("---")
st.markdown("© 2025 Afriland First Bank - Application de Prédiction des Défauts de Paiement") 

# git add .
# git commit -m "Mes nouvelles modifications"
# git push
#pipreqs . --force