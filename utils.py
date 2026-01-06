
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import ccxt
from datetime import datetime, timezone
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch_ohlcv_yahoo(pair: str, timeframe: str, start_date: str, end_date: str, max_retries: int = 3, retry_delay: int = 5):
    """
    Récupère les données OHLCV depuis Yahoo Finance avec gestion des rate limits.
    Fonctionne bien depuis Streamlit Cloud car Yahoo Finance n'a pas de restrictions géographiques.
    
    Args:
        pair: Paire de trading (ex: "BTCUSDC" -> converti en "BTC-USD")
        timeframe: Période (ex: "1h")
        start_date: Date de début (format string "YYYY-MM-DD")
        end_date: Date de fin (format string "YYYY-MM-DD")
        max_retries: Nombre max de tentatives en cas de rate limit (défaut: 3)
        retry_delay: Délai en secondes entre les tentatives (défaut: 5)
    
    Returns:
        DataFrame avec les colonnes Timestamp, Open, High, Low, Close, Volume
    """
    import time
    
    try:
        import yfinance as yf
    except ImportError:
        raise Exception("yfinance n'est pas installé. Installez-le avec: pip install yfinance")
    
    # Conversion de la paire : BTCUSDC -> BTC-USD (Yahoo Finance n'a pas BTC/USDC direct)
    # BTC-USD est très proche de BTC/USDC car USDC est indexé sur USD
    if "BTC" in pair.upper():
        ticker = "BTC-USD"
    else:
        # Fallback pour d'autres paires
        ticker = pair.replace("USDC", "-USD").replace("USDT", "-USD")
    
    # Conversion du timeframe
    # yfinance accepte: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    interval_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
    }
    interval = interval_map.get(timeframe, "1h")
    
    logger.info(f"Récupération des données depuis Yahoo Finance (ticker={ticker}, interval={interval}, start={start_date}, end={end_date})")
    
    # Tentative avec retry pour gérer les rate limits
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # Créer l'objet ticker
            ticker_obj = yf.Ticker(ticker)
            
            # Pour les données récentes, Yahoo Finance fonctionne mieux avec 'period' qu'avec 'start'/'end'
            # Pour les intervalles intraday (1h, etc.), utiliser 'period' pour obtenir les données les plus récentes
            # period peut être: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            # Calculer le nombre de jours nécessaire
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            days_diff = (end_dt - start_dt).days
            
            # Pour les périodes courtes (≤ 60 jours), utiliser period pour obtenir les données les plus récentes
            # C'est particulièrement important pour les données intraday récentes (1h, etc.)
            # Yahoo Finance avec 'period' retourne les données jusqu'à maintenant, ce qui est mieux pour les données récentes
            if days_diff <= 60:
                # Utiliser period avec un peu plus de jours pour être sûr d'avoir toutes les données
                # Pour 7 jours, on demande 8-10 jours pour avoir les données les plus récentes
                period_days = max(7, days_diff + 3)  # Au moins 7 jours, ou jours demandés + 3
                period = f"{period_days}d"
                logger.info(f"Utilisation de period={period} pour récupérer les données récentes (interval={interval})")
                df = ticker_obj.history(period=period, interval=interval)
                
                # Filtrer les données pour la période demandée (mais garder les données récentes)
                if not df.empty:
                    start_dt_utc = pd.to_datetime(start_date, utc=True)
                    end_dt_utc = pd.to_datetime(end_date, utc=True)
                    # S'assurer que les timestamps sont en UTC pour la comparaison
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    elif df.index.tz != timezone.utc:
                        df.index = df.index.tz_convert('UTC')
                    
                    # Filtrer par date de début, mais garder toutes les données jusqu'à maintenant
                    # (ne pas filtrer par end_date pour garder les données les plus récentes)
                    current_time_utc = pd.to_datetime(datetime.now(timezone.utc), utc=True)
                    df = df[(df.index >= start_dt_utc) & (df.index <= current_time_utc)]
                    logger.info(f"Données filtrées: {len(df)} lignes de {df.index.min()} à {df.index.max()}")
            else:
                # Pour les périodes plus longues, utiliser start/end
                logger.info(f"Utilisation de start/end pour période longue ({days_diff} jours)")
                df = ticker_obj.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise Exception(f"Aucune donnée récupérée depuis Yahoo Finance pour {ticker}")
            
            # Renommer et réorganiser les colonnes pour correspondre au format attendu
            df = df.reset_index()
            df = df.rename(columns={
                "Datetime": "Timestamp",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            })
            
            # Si la colonne s'appelle "Date" au lieu de "Datetime"
            if "Date" in df.columns and "Timestamp" not in df.columns:
                df = df.rename(columns={"Date": "Timestamp"})
            
            # S'assurer que Timestamp est en UTC
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
            else:
                raise Exception("Colonne Timestamp introuvable dans les données Yahoo Finance")
            
            # Sélectionner uniquement les colonnes nécessaires
            required_cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            df = df[required_cols].copy()
            
            # Trier par timestamp
            df = df.sort_values("Timestamp").reset_index(drop=True)
            
            logger.info(f"✅ Données récupérées depuis Yahoo Finance: {len(df)} lignes")
            return df
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Vérifier si c'est un rate limit error
            # yfinance peut lever YFRateLimitError ou des exceptions génériques avec "rate limit" dans le message
            is_rate_limit = (
                "rate limit" in error_msg.lower() or 
                "YFRateLimitError" in error_type or 
                "429" in error_msg or
                "Too Many Requests" in error_msg
            )
            
            if is_rate_limit and retry_count < max_retries - 1:
                retry_count += 1
                wait_time = retry_delay * retry_count  # Backoff linéaire (5s, 10s, 15s)
                logger.warning(f"⚠️ Rate limit détecté (tentative {retry_count}/{max_retries}). Attente de {wait_time}s avant de réessayer...")
                time.sleep(wait_time)
                last_error = e
                continue
            else:
                # Autre erreur ou max retries atteint
                if is_rate_limit:
                    logger.error(f"❌ Rate limit persistant après {max_retries} tentatives: {error_type}: {error_msg}")
                else:
                    logger.error(f"❌ Erreur lors de la récupération depuis Yahoo Finance: {error_type}: {error_msg}")
                raise Exception(f"Erreur Yahoo Finance: {error_type}: {error_msg}")
    
    # Si on arrive ici, toutes les tentatives ont échoué
    raise Exception(f"Échec après {max_retries} tentatives. Dernière erreur: {type(last_error).__name__}: {str(last_error)}")

def fetch_ohlcv_from_csv(csv_path: str, start_date: str, end_date: str):
    """
    Récupère les données OHLCV depuis un fichier CSV local (fallback).
    
    Args:
        csv_path: Chemin vers le fichier CSV
        start_date: Date de début (format string)
        end_date: Date de fin (format string)
    
    Returns:
        DataFrame avec les colonnes Timestamp, Open, High, Low, Close, Volume
    """
    try:
        df = pd.read_csv(csv_path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        
        # Filtrer par date
        df_filtered = df[(df["Timestamp"] >= start_dt) & (df["Timestamp"] <= end_dt)]
        
        if len(df_filtered) == 0:
            # Si aucune donnée dans la plage, prendre les dernières données disponibles
            df_filtered = df.tail(168)  # 7 jours * 24h = 168 heures
        
        df_filtered = df_filtered.sort_values("Timestamp").reset_index(drop=True)
        return df_filtered
    
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du CSV: {str(e)}")

def fetch_ohlcv_binance(pair: str, timeframe: str, start_date: int, end_date: int, limit=1000, timeout=30, max_retries=3):
    """
    Récupère les données OHLCV depuis Binance avec gestion d'erreur et timeout.
    
    Args:
        pair: Paire de trading (ex: "BTCUSDC")
        timeframe: Période (ex: "1h")
        start_date: Date de début (format string ou datetime)
        end_date: Date de fin (format string ou datetime)
        limit: Nombre max de candles par requête (défaut: 1000)
        timeout: Timeout en secondes pour chaque requête (défaut: 30)
        max_retries: Nombre max de tentatives en cas d'échec (défaut: 3)
    
    Returns:
        DataFrame avec les colonnes Timestamp, Open, High, Low, Close, Volume
    """
    import time
    
    # Configuration de ccxt avec timeout et options pour Streamlit Community
    logger.info(f"Initialisation de l'exchange Binance avec timeout={timeout}s, max_retries={max_retries}")
    try:
        ex = ccxt.binance({
            "enableRateLimit": True,
            "timeout": timeout * 1000,  # timeout en millisecondes
            "options": {
                "defaultType": "spot",  # Utiliser le marché spot
            },
            "rateLimit": 1200,  # Limite de rate (ms entre requêtes)
        })
        logger.info("Exchange Binance initialisé avec succès")
        
        # Essayer de charger les markets manuellement avec gestion d'erreur
        # Cela évite que load_markets() soit appelé automatiquement lors du premier fetch_ohlcv
        try:
            logger.info("Tentative de chargement des markets...")
            ex.load_markets()
            logger.info("Markets chargés avec succès")
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, 
                ccxt.BaseError, Exception) as market_error:
            # Si le chargement des markets échoue, on continue quand même
            # car certaines paires peuvent fonctionner sans markets chargés
            logger.warning(f"Échec du chargement des markets (continuons quand même): {type(market_error).__name__}: {str(market_error)}")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de l'exchange Binance: {type(e).__name__}: {str(e)}")
        raise Exception(f"Erreur lors de l'initialisation de l'exchange Binance: {str(e)}")
    
    all_rows = []
    since = to_ms(pd.to_datetime(start_date))
    end_ms = to_ms(pd.to_datetime(end_date))
    
    last_error = None
    
    while since < end_ms:
        retry_count = 0
        batch = None
        
        # Tentative de récupération avec retry pour chaque batch
        while retry_count < max_retries:
            try:
                # Tentative de récupération des données
                # Note: load_markets() est appelé automatiquement par fetch_ohlcv
                logger.info(f"Tentative {retry_count + 1}/{max_retries} de récupération des données pour since={since}")
                batch = ex.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)
                logger.info(f"Données récupérées avec succès: {len(batch)} candles")
                break  # Succès, sortir de la boucle de retry
                
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, 
                    ccxt.BaseError) as e:
                # Erreur réseau (timeout, connexion, exchange non disponible, etc.)
                retry_count += 1
                last_error = e
                error_type = type(e).__name__
                error_msg = str(e)
                logger.warning(f"Erreur réseau/exchange (tentative {retry_count}/{max_retries}): {error_type}: {error_msg}")
                if retry_count < max_retries:
                    # Attendre avant de réessayer (backoff exponentiel)
                    wait_time = min(2 ** retry_count, 10)
                    logger.info(f"Attente de {wait_time}s avant de réessayer...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Échec après {max_retries} tentatives: {error_type}: {error_msg}")
                    raise Exception(f"Erreur réseau/exchange après {max_retries} tentatives pour la période {since}: {error_type}: {error_msg}")
            
            except ccxt.ExchangeError as e:
                # Erreur de l'exchange (rate limit, etc.)
                retry_count += 1
                last_error = e
                error_type = type(e).__name__
                error_msg = str(e)
                logger.warning(f"Erreur de l'exchange (tentative {retry_count}/{max_retries}): {error_type}: {error_msg}")
                if retry_count < max_retries:
                    wait_time = min(2 ** retry_count, 10)
                    logger.info(f"Attente de {wait_time}s avant de réessayer...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Échec après {max_retries} tentatives: {error_type}: {error_msg}")
                    raise Exception(f"Erreur de l'exchange après {max_retries} tentatives pour la période {since}: {error_type}: {error_msg}")
            
            except Exception as e:
                # Autre erreur - ne pas retry
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"Erreur inattendue: {error_type}: {error_msg}")
                raise Exception(f"Erreur inattendue lors de la récupération des données: {error_type}: {error_msg}")
        
        if not batch:
            break
        
        all_rows.extend(batch)
        # avance d'un pas après le dernier timestamp
        since = batch[-1][0] + 1
        
        # sécurité anti-boucle
        if len(batch) < 10:
            break
    
    if not all_rows:
        raise Exception(f"Aucune donnée récupérée. Dernière erreur: {last_error}")
    
    df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df

def fetch_ohlcv_binance_with_fallback(pair: str, timeframe: str, start_date: str, end_date: str, 
                                       csv_path: str = "btc_usdc_1h_2015_2025.csv", 
                                       timeout: int = 10, max_retries: int = 2):
    """
    Récupère les données OHLCV avec fallback multi-niveaux :
    1. API Binance (si disponible)
    2. Yahoo Finance (si Binance échoue - fonctionne depuis Streamlit Cloud)
    3. CSV local (dernier recours)
    
    Optimisé pour Streamlit Community avec timeout court et retries limités.
    
    Args:
        pair: Paire de trading (ex: "BTCUSDC")
        timeframe: Période (ex: "1h")
        start_date: Date de début (format string)
        end_date: Date de fin (format string)
        csv_path: Chemin vers le fichier CSV de fallback
        timeout: Timeout en secondes (défaut: 10 pour Streamlit Community)
        max_retries: Nombre max de tentatives (défaut: 2)
    
    Returns:
        Tuple (DataFrame, str): DataFrame avec les colonnes Timestamp, Open, High, Low, Close, Volume
                               et la source des données ("API Binance", "Yahoo Finance" ou "CSV local")
    """
    import os
    
    logger.info(f"Tentative de récupération des données (pair={pair}, timeframe={timeframe}, start={start_date}, end={end_date})")
    
    # 1. Tentative de récupération via API Binance
    try:
        logger.info("🔄 Tentative 1/3: API Binance...")
        df = fetch_ohlcv_binance(pair, timeframe, start_date, end_date, 
                                 timeout=timeout, max_retries=max_retries)
        logger.info(f"✅ Données récupérées avec succès depuis l'API Binance: {len(df)} lignes")
        return df, "API Binance"
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.ExchangeNotAvailable, 
            ccxt.RequestTimeout, ccxt.BaseError, Exception) as api_error:
        error_type = type(api_error).__name__
        error_msg = str(api_error)
        logger.warning(f"❌ Échec de l'API Binance: {error_type}: {error_msg}")
        
        # 2. Tentative de récupération via Yahoo Finance (fonctionne depuis Streamlit Cloud)
        try:
            logger.info("🔄 Tentative 2/3: Yahoo Finance...")
            # Ajouter un petit délai avant d'essayer Yahoo Finance pour éviter les rate limits
            import time
            time.sleep(2)  # Délai de 2 secondes pour éviter les rate limits
            df = fetch_ohlcv_yahoo(pair, timeframe, start_date, end_date, max_retries=3, retry_delay=10)
            logger.info(f"✅ Données récupérées avec succès depuis Yahoo Finance: {len(df)} lignes")
            return df, "Yahoo Finance"
        except Exception as yahoo_error:
            error_type_yahoo = type(yahoo_error).__name__
            error_msg_yahoo = str(yahoo_error)
            logger.warning(f"❌ Échec de Yahoo Finance: {error_type_yahoo}: {error_msg_yahoo}")
            
            # 3. Dernier recours : CSV local
            logger.info(f"🔄 Tentative 3/3: Fichier CSV local ({csv_path})...")
            if os.path.exists(csv_path):
                try:
                    df = fetch_ohlcv_from_csv(csv_path, start_date, end_date)
                    logger.info(f"✅ Données récupérées depuis le CSV: {len(df)} lignes")
                    return df, "CSV local"
                except Exception as csv_error:
                    logger.error(f"❌ Erreur lors de la lecture du CSV: {type(csv_error).__name__}: {str(csv_error)}")
                    raise Exception(
                        f"Toutes les sources ont échoué:\n"
                        f"- Binance: {error_type}: {error_msg}\n"
                        f"- Yahoo Finance: {error_type_yahoo}: {error_msg_yahoo}\n"
                        f"- CSV: {type(csv_error).__name__}: {str(csv_error)}"
                    )
            else:
                logger.error(f"❌ Fichier CSV non trouvé: {csv_path}")
                raise Exception(
                    f"Toutes les sources ont échoué et le CSV est introuvable:\n"
                    f"- Binance: {error_type}: {error_msg}\n"
                    f"- Yahoo Finance: {error_type_yahoo}: {error_msg_yahoo}\n"
                    f"- CSV: Fichier non trouvé ({csv_path})"
                )

def plot_backtest(backtester, plot=True):
    # On suppose que trades_df == backtester.df_trades déjà généré avec l'algo ci-dessus
    trades_df = backtester.df_trades

    # Pour le graphique, récupérer le temps et close price
    df_curves = backtester.df_bt.reset_index(drop=True)
    df_curves["Timestamp_entry"] = df_curves["Timestamp"]
    
    # Vérifier si trades_df est vide ou n'a pas les colonnes nécessaires
    if len(trades_df) > 0 and "Timestamp" in trades_df.columns and "exit_price" in trades_df.columns and "Capital" in trades_df.columns:
        df_curves = pd.merge(df_curves, trades_df[["Timestamp", "exit_price","Capital"]], on="Timestamp", how="left")
    else:
        df_curves["exit_price"] = None
        df_curves["Capital"] = backtester.capital_init
    
    if len(trades_df) > 0 and "Timestamp_entry" in trades_df.columns and "entry_price" in trades_df.columns:
        df_curves = pd.merge(df_curves, trades_df[["Timestamp_entry", "entry_price"]], on="Timestamp_entry", how="left")
    else:
        df_curves["entry_price"] = None
    
    df_curves["Capital"] = df_curves["Capital"].ffill().fillna(backtester.capital_init)

    timestamps = df_curves["Timestamp"]
    close_prices = df_curves["Close"]
    capital_curve = df_curves["Capital"]
    buy_time = df_curves["Timestamp_entry"]
    buy_price = df_curves["entry_price"]
    sell_time = df_curves["Timestamp"]
    sell_price = df_curves["exit_price"]

    # Créer un subplot avec 2 graphiques (prix en haut, capital en bas)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Cours Close avec signaux Buy/Sell', 'Évolution du Capital'),
        row_heights=[0.6, 0.4]
    )

    # Graphique 1 : Prix avec signaux
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=close_prices,
            mode='lines',
            name='Close',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=buy_time,
            y=buy_price,
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sell_time,
            y=sell_price,
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell'
        ),
        row=1, col=1
    )

    # Graphique 2 : Capital
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=capital_curve,
            mode='lines',
            name='Capital',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )

    # Ligne de référence pour le capital initial
    fig.add_hline(
        y=backtester.capital_init,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Capital initial: {backtester.capital_init:.2f}",
        row=2, col=1
    )

    fig.update_layout(
        title='Cours Close avec signaux Buy/Sell et Évolution du Capital',
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(x=0, y=1)
    )

    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(title_text="Capital", row=2, col=1)

    if plot:
        fig.show()
    return fig

def plot_predictions(df_bt, signal, plot=True):

    # Pour le graphique, récupérer le temps et close price
    df_curves = df_bt.reset_index(drop=True)
    df_curves["Timestamp_entry"] = df_curves["Timestamp"]
    
    # Vérifier si trades_df est vide ou n'a pas les colonnes nécessaires
    #df_curves = pd.merge(df_curves, trades_df[["Timestamp", "exit_price","Capital"]], on="Timestamp", how="left")
    if signal == 1:
        df_curves.loc[df_curves.tail(1).index,"Timestamp"] = df_bt["Timestamp"].iloc[-1]
        df_curves.loc[df_curves.tail(1).index,"entry_price"] = df_bt["Close"].iloc[-1]
        buy_price = df_curves["entry_price"]
        sell_price = None

    else:
        df_curves.loc[df_curves.tail(1).index,"Timestamp"] = df_bt["Timestamp"].iloc[-1]
        df_curves.loc[df_curves.tail(1).index,"exit_price"] = df_bt["Close"].iloc[-1]
        sell_price = df_curves["exit_price"]
        buy_price = None

    timestamps = df_curves["Timestamp"]
    close_prices = df_curves["Close"]
    #ajouter un point 24h après la dernière ligne et relier le prix de ce point en pointillés au dernier prix
    future_list = []
    future_list.append(pd.DataFrame({"Timestamp": [df_curves["Timestamp"].iloc[-1]], "Close": [df_curves["Close"].iloc[-1]]}))
    future_list.append(pd.DataFrame({"Timestamp": [df_curves["Timestamp"].iloc[-1] + pd.Timedelta(hours=24)], "Close": [df_curves["Close"].iloc[-1] * (1 + 0.002*(2*signal-1))]}))

    future_price = pd.concat(future_list)
    print(f"signal: {signal}")
    print(df_curves["Close"])
    print(f"future_price: {future_price}")

    buy_time = df_curves["Timestamp_entry"]
    sell_time = df_curves["Timestamp"]
    # Créer un subplot avec 1 graphique
    fig = go.Figure()

    # Graphique 1 : Prix avec signaux Future Price
    if signal == 1:
        fig.add_trace(
            go.Scatter(
                x=future_price["Timestamp"],
                y=future_price["Close"],
                mode='lines',
                name='Future Price',
                line=dict(color='green', dash='dot')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=buy_time,
                y=buy_price,
                mode='markers',
                marker=dict(color='green', symbol='triangle-up', size=10),
                name='Buy'
            )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=future_price["Timestamp"],
                y=future_price["Close"],
                mode='lines',
                name='Future Price',
                line=dict(color='red', dash='dot')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sell_time,
                y=sell_price,
                mode='markers',
                marker=dict(color='red', symbol='triangle-down', size=10),
                name='Sell'
            ),
        )        


    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=close_prices,
            mode='lines',
            name='Close',
            line=dict(color='blue')
        ),
    )

    fig.update_layout(
        title='Cours Close avec signaux Buy/Sell',
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(x=0, y=1)
    )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Prix")

    if plot:
        fig.show()
    return fig

def clean_data(df):
    df = df.copy()
    df = df.dropna().reset_index(drop=True)
    return df

def calculate_features_pct_change(df):
    df = df.copy()
    df["Close_pct_change"] = df["Close"].pct_change()
    df["High_pct_change"] = df["High"].pct_change()
    df["Low_pct_change"] = df["Low"].pct_change()          
    df["Volume_pct_change"] = df["Volume"].pct_change()
    features_cols = ["Close_pct_change", "High_pct_change", "Low_pct_change", "Volume_pct_change"]
    return df, features_cols

def calculate_features_technical(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low  = df["Low"]
    vol  = df["Volume"]
    open = df["Open"]

    df["logret_1"] = np.log(close).diff()
    df["logret_5"] = np.log(close).diff(5)
    df["logret_20"] = np.log(close).diff(20)

    df["vol_20"] = df["logret_1"].rolling(20).std()
    df["vol_50"] = df["logret_1"].rolling(50).std()

    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    df["ma_diff"] = df["ma20"] - df["ma50"]
    df["ema_diff"] = df["ema20"] - df["ema50"]

    df["rsi14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["atr14"] = ta.volatility.average_true_range(high, low, close, window=14) / close
    df["adx14"] = ta.trend.adx(high, low, close, window=14)

    df["hl_range"] = (high - low) / close
    df["np_range"] = (high - low) / (close - open+1e-6)
    features_cols = []

    # On retire les lignes avec NaNs (features + futur)
    features_cols = ["logret_1", "logret_5", "logret_20", "vol_20", 
    "vol_50", "ma20", "ma50", "ema20", "ema50", "ma_diff", "ema_diff", 
    "rsi14", "macd", "macd_signal", "atr14", "adx14", "hl_range", "np_range"]
    
    return df, features_cols

def calculate_label(df, horizon_steps, threshold):
    df = df.copy()
    # --- Label : ROI futur à horizon_steps ---
    df["future_close"] = df["Close"].shift(-horizon_steps)
    df["roi_H"] = (df["future_close"] - df["Close"]) / df["Close"]
    df["y"] = (df["roi_H"] > threshold).astype(int)
    return df

def prepare_data_min_features(df,horizon_steps,threshold):
    # Nettoyage basique
    df = clean_data(df)
    df, features_cols = calculate_features_pct_change(df)
    df = calculate_label(df,horizon_steps,threshold)
    df_model = df.dropna(subset=features_cols + ["y"]).reset_index(drop=True)
    df_model["Volume_pct_change"] = df_model["Volume_pct_change"].replace([np.inf, -np.inf], 0)

    return df_model, features_cols


def prepare_data_advanced_features(df,horizon_steps,threshold):
    # Nettoyage basique
    df = clean_data(df)
    df,features_cols = calculate_features_technical(df)
    df = calculate_label(df,horizon_steps,threshold)
    df_model = df.dropna(subset=features_cols + ["y"]).reset_index(drop=True)
    return df_model, features_cols
    