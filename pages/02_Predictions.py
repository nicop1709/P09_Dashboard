from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pandas import to_datetime
import numpy as np
from backtest import Backtest
from datetime import datetime
from utils import fetch_ohlcv_binance_with_fallback, prepare_data_advanced_features, plot_predictions
from sklearn.preprocessing import StandardScaler
import joblib

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
            timeout=5,  # Timeout très court pour Streamlit Community (5 secondes)
            max_retries=1  # Une seule tentative pour éviter les timeouts longs
        )
    except Exception as e:
        # Si même le fallback échoue, afficher l'erreur
        st.error(f"❌ Erreur lors de la récupération des données: {str(e)}")
        st.info("💡 **Conseil**: Vérifiez votre connexion internet ou que le fichier CSV est présent dans le projet.")
        st.stop()
    
    # Afficher la source des données
    if data_source == "API Binance":
        st.success("✅ Données récupérées depuis l'API Binance")
    else:
        st.warning("⚠️ Données récupérées depuis le fichier CSV local (API Binance indisponible)")
    
    progress_bar.progress(0.3, text="Préparation des données")
    last_data_model, features_cols_model = prepare_data_advanced_features(last_data, 24, 0.002)
    
    progress_bar.progress(0.7, text="Normalisation des données")
    last_data_model_s = scaler.transform(last_data_model[features_cols_model].values)
    
    progress_bar.progress(0.8, text="Prédiction TabNet")
    tabnet_pred_tab = tabnet_model.predict_proba(last_data_model_s)[:, 1]
    
    progress_bar.progress(1.0, text="Fin de la prédiction")
    tabnet_pred = (tabnet_pred_tab >= 0.5).astype(int)
    
    # Afficher les informations
    st.info(f"Le cours du Bitcoin est actuellement de ***{last_data_model['Close'].iloc[-1]:.2f} USDC***")
    
    # Vérifier si les données sont récentes
    last_timestamp = last_data_model['Timestamp'].iloc[-1]
    current_time = to_datetime(datetime.utcnow(), utc=True)
    time_diff_minutes = (current_time - last_timestamp).total_seconds() / 60
    
    if time_diff_minutes < 5:
        st.success(f"Le signal TabNet pour les 5 premières minutes de l'heure est : ***{'Buy' if tabnet_pred[-1] == 1 else 'No-trade or Sell'}***")
        fig = plot_predictions(last_data, tabnet_pred[-1], plot=False)
        st.plotly_chart(fig)
    else:
        wait_minutes = round((last_timestamp + timedelta(minutes=60) - current_time).total_seconds() / 60)
        st.warning(f"Le signal TabNet n'est pas disponible pour trader, il faut attendre **{wait_minutes} minutes**")
        st.warning(f"Si la tendance ne change pas le signal est actuellement : {'***Buy***' if tabnet_pred[-1] == 1 else 'No-trade or Sell'}")
        fig = plot_predictions(last_data, tabnet_pred[-1], plot=False)
        st.plotly_chart(fig)
        
except Exception as e:
    st.error(f"❌ Erreur lors du traitement: {str(e)}")
    st.exception(e)
    st.stop()