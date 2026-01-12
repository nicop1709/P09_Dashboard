from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pandas import to_datetime
import numpy as np
from backtest import Backtest
from datetime import datetime, timezone
from utils import fetch_ohlcv_binance_with_fallback, prepare_data_advanced_features, plot_predictions
from sklearn.preprocessing import StandardScaler
import joblib

# Chargement des modèles pré-entraînés
# Note: Les warnings InconsistentVersionWarning sont normaux si le modèle a été sauvegardé
# avec une version antérieure de scikit-learn. La compatibilité ascendante est généralement assurée.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    scaler = joblib.load("models/tabnet_scaler.pkl")
    tabnet_model = joblib.load("models/tabnet_model.pkl")

st.set_page_config(page_icon=":moneybag:")

st.title("Predictions")

st.write("""Ce notebook prédit les signaux de trading pour la paire BTC-USDC 1h de 2015 à 2025.
Il utilise les modèles RandomForest et TabNet pour prédire les signaux de trading.
Il affiche les statistiques des performances des modèles.
Il affiche le backtest des modèles.
Il affiche la liste des trades des modèles.
""")


st.header("Predictions")

st.write(f"Nous sommes le {format(datetime.now(), '%d/%m/%Y')},  il est {format(datetime.now(), '%H:%M:%S')}")

progress_bar = st.progress(0.0)

try:
    progress_bar.progress(0.1, text="Récupération des données")
    
    # Utilisation de la fonction avec fallback (timeout court pour Streamlit Community)
    try:
        last_data, data_source = fetch_ohlcv_binance_with_fallback(
            "BTCUSDC", 
            "1h", 
            format(datetime.now()-timedelta(days=7),"%Y-%m-%d"), 
            format(datetime.now(),"%Y-%m-%d"),
            timeout=15,  # Timeout pour Streamlit Community (15 secondes - compromis entre vitesse et fiabilité)
            max_retries=2  # Deux tentatives pour plus de robustesse
        )
    except Exception as e:
        # Si même le fallback échoue, afficher l'erreur
        st.error(f"❌ Erreur lors de la récupération des données: {str(e)}")
        st.stop()
    
    progress_bar.progress(0.3, text="Préparation des données")
    last_data_model, features_cols_model = prepare_data_advanced_features(last_data, 24, 0.002)
    
    progress_bar.progress(0.7, text="Normalisation des données")
    last_data_model_s = scaler.transform(last_data_model[features_cols_model].values)
    
    progress_bar.progress(0.8, text="Prédiction TabNet")
    tabnet_pred_tab = tabnet_model.predict_proba(last_data_model_s)[:, 1]
    
    progress_bar.progress(0.9, text="Calcul de l'explicabilité locale")
    # Calculer l'explicabilité pour la dernière prédiction
    last_sample = last_data_model_s[-1:]  # Prendre la dernière ligne (déjà en format 2D)
    try:
        tabnet_explain = tabnet_model.explain(last_sample)

        # Si explain() retourne un tuple, prendre le premier élément (masks ou importance)
        if isinstance(tabnet_explain, tuple):
            tabnet_explain = tabnet_explain[0]
        # Convertir en numpy array si nécessaire
        if not isinstance(tabnet_explain, np.ndarray):
            tabnet_explain = np.array(tabnet_explain)
    except Exception as explain_error:
        st.warning(f"⚠️ Impossible de calculer l'explicabilité: {str(explain_error)}")
        tabnet_explain = None
    
    progress_bar.progress(1.0, text="Fin de la prédiction")
    tabnet_pred = (tabnet_pred_tab >= 0.5).astype(int)
    
    # Afficher les informations
    st.info(f"Le cours du Bitcoin est actuellement de ***{last_data_model['Close'].iloc[-1]:.2f} USDC***")
    
    # Vérifier si les données sont récentes
    last_timestamp = last_data_model['Timestamp'].iloc[-1]
    current_time = to_datetime(datetime.now(timezone.utc), utc=True)
    time_diff_minutes = (current_time - last_timestamp).total_seconds() / 60
    
    # Calculer le temps jusqu'à la prochaine heure complète
    if time_diff_minutes < 5:
        # Données très récentes (< 5 minutes)
        st.success(f"Le signal TabNet pour les 5 premières minutes de l'heure est : ***{'Buy' if tabnet_pred[-1] == 1 else 'No-trade or Sell'}***")
        fig = plot_predictions(last_data, tabnet_pred[-1], plot=False)
        st.plotly_chart(fig)
    elif time_diff_minutes < 60:
        # Données dans la dernière heure : calculer le temps jusqu'à la prochaine heure complète
        # Calculer l'heure suivante après last_timestamp
        next_hour = last_timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        # Si next_hour est dans le passé, prendre l'heure suivante
        if next_hour <= current_time:
            next_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        wait_minutes = round((next_hour - current_time).total_seconds() / 60)
        wait_minutes = max(0, min(60, wait_minutes))  # Limiter entre 0 et 60 minutes
        
        st.warning(f"Le signal TabNet n'est pas disponible pour trader, il faut attendre **{wait_minutes} minutes**")
        st.warning(f"Si la tendance ne change pas le signal est actuellement : {'***Buy***' if tabnet_pred[-1] == 1 else 'No-trade or Sell'}")
        fig = plot_predictions(last_data, tabnet_pred[-1], plot=False)
        st.plotly_chart(fig)
    else:
        # Données anciennes (plus d'une heure) - ne pas calculer wait_minutes
        st.warning(f"Les données sont anciennes ({int(time_diff_minutes)} minutes). Le signal n'est pas disponible pour trader.")
        st.warning(f"Signal basé sur les dernières données disponibles : {'***Buy***' if tabnet_pred[-1] == 1 else 'No-trade or Sell'}")
        fig = plot_predictions(last_data, tabnet_pred[-1], plot=False)
        st.plotly_chart(fig)
    
    # Graphique d'explicabilité locale
    if tabnet_explain is not None:
        st.header("Explicabilité locale de la prédiction TabNet")
        st.write("Ce graphique montre l'importance de chaque feature pour la dernière prédiction.")
        
        # Extraire l'importance des features pour la dernière prédiction
        # tabnet_explain retourne un array de shape (n_samples, n_features)
        feature_importance = tabnet_explain[0] if len(tabnet_explain.shape) > 1 else tabnet_explain  # Prendre la première (et seule) prédiction
        
        # Créer un DataFrame pour faciliter la visualisation
        importance_df = pd.DataFrame({
            'Feature': features_cols_model,
            'Importance': feature_importance
        })
        
        # Trier par valeur absolue de l'importance
        importance_df['Abs_Importance'] = importance_df['Importance'].abs()
        importance_df = importance_df.sort_values('Abs_Importance', ascending=True)
        
        # Créer le graphique en barres horizontal
        fig_explain = go.Figure()
        
        # Couleurs selon le signe de l'importance
        colors = ['green' if x > 0 else 'red' for x in importance_df['Importance']]
        
        fig_explain.add_trace(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{x:.4f}" for x in importance_df['Importance']],
            textposition='outside',
            name='Importance'
        ))
        
        fig_explain.update_layout(
            title='Importance des features pour la prédiction locale (TabNet)',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600,
            width=1000,
            showlegend=False,
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
        )
        
        st.plotly_chart(fig_explain, use_container_width=True)
    

except Exception as e:
    st.error(f"❌ Erreur lors du traitement: {str(e)}")
    st.exception(e)
    st.stop()