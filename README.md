# Projet 9 - Dashboard Trading BTC/USDC

Un dashboard interactif pour comparer les performances de modèles de machine learning (RandomForest vs TabNet) sur la prédiction de signaux de trading pour la paire BTC-USDC.

## 🎯 Ce que fait ce projet

Ce projet permet de :
- Explorer les données historiques BTC-USDC depuis 2015
- Visualiser les performances théoriques d'un backtest "parfait" (avec connaissance du futur)
- Obtenir des prédictions en temps réel via le modèle TabNet entraîné
- Comparer différentes stratégies de trading avec calcul automatique du ROI, drawdown, win rate, etc.

Le dashboard est construit avec Streamlit et propose deux pages principales : une pour l'analyse exploratoire des données (EDA) et une autre pour les prédictions en temps réel.

## 📋 Prérequis

- Python 3.8 ou supérieur
- Un fichier de données historiques `btc_usdc_1h_2015_2025.csv` (pour la page EDA)
- Les modèles pré-entraînés dans le dossier `models/` :
  - `tabnet_model.pkl`
  - `tabnet_scaler.pkl`

## 🚀 Installation

1. Clonez le repository (ou téléchargez les fichiers)

2. Créez un environnement virtuel (recommandé) :
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

**Note** : L'installation de PyTorch (requis pour TabNet) peut prendre plusieurs minutes car c'est un package volumineux. La version CPU standard est suffisante pour ce projet. Si vous avez un GPU NVIDIA et souhaitez l'utiliser, vous pouvez installer une version spécifique avec support CUDA depuis [pytorch.org](https://pytorch.org/get-started/locally/).

## 💻 Utilisation

Lancez le dashboard avec :
```bash
streamlit run Home.py
```

Le dashboard s'ouvrira automatiquement dans votre navigateur (généralement sur `http://localhost:8501`).

### Pages disponibles

**Home** : Page d'accueil avec la description du projet et les liens vers les autres pages

**EDA** : 
- Visualisation des données historiques avec graphiques candlestick
- Calcul des features et de la target
- Statistiques descriptives
- Backtest "parfait" avec réinjection de capital
- Ajustement des paramètres via la sidebar (nombre de jours affichés, horizon de prédiction, fees)

**Predictions** :
- Récupération automatique des dernières données Binance (7 derniers jours)
- Prédiction en temps réel avec le modèle TabNet
- Affichage du signal de trading (Buy / No-trade or Sell)
- Graphique avec projection du prix futur attendu

## 📁 Structure du projet

```
P09_Dashboard/
├── Home.py                 # Page d'accueil Streamlit
├── backtest.py             # Classe Backtest pour simuler les stratégies
├── trader.py               # Classe Trader pour gérer les positions
├── utils.py                # Fonctions utilitaires (fetch données, features, graphiques)
├── requirements.txt        # Dépendances Python
├── pages/
│   ├── 01_EDA.py          # Page d'analyse exploratoire
│   └── 02_Predictions.py   # Page de prédictions en temps réel
└── models/
    ├── tabnet_model.pkl    # Modèle TabNet pré-entraîné
    └── tabnet_scaler.pkl   # Scaler pour normaliser les features
```

## 🔧 Fonctionnalités principales

### Backtest
Le système de backtest simule une stratégie de trading avec :
- Réinjection de capital (compound interest)
- Prise en compte des fees de trading (roundtrip)
- Calcul automatique du ROI annualisé, max drawdown, win rate
- Liste détaillée de tous les trades

### Features techniques
Le modèle utilise des features avancées calculées avec la librairie `ta` :
- Retours logarithmiques (1h, 5h, 20h)
- Volatilité (rolling std sur 20 et 50 périodes)
- Moyennes mobiles (MA20, MA50, EMA20, EMA50)
- Indicateurs techniques : RSI, MACD, ATR, ADX
- Ratios de range (high-low, etc.)

### Prédictions
Les prédictions sont générées pour un horizon de 24 heures (24 bougies de 1h). Le modèle prédit si le ROI futur dépassera un seuil de 0.2% (après déduction des fees).

## ⚠️ Notes importantes

- Les données historiques pour l'EDA doivent être dans un fichier `btc_usdc_1h_2015_2025.csv` à la racine du projet
- Les prédictions nécessitent une connexion internet pour récupérer les données Binance
- Le modèle TabNet doit être pré-entraîné (pas d'entraînement dans ce dashboard)
- Les résultats de backtest sont des simulations et ne garantissent pas les performances futures
- **PyTorch et pytorch-tabnet sont requis** : Assurez-vous que ces packages sont bien installés, sinon vous obtiendrez une erreur `ModuleNotFoundError: No module named 'pytorch_tabnet'` lors du chargement du modèle

## 📊 Métriques calculées

- **ROI** : Retour sur investissement total
- **ROI annualisé** : ROI projeté sur une année
- **ROI par jour** : ROI moyen quotidien
- **Max DrawDown** : Perte maximale depuis un pic
- **Win rate** : Pourcentage de trades gagnants
- **Nombre de trades** : Total et par jour

## 🛠️ Technologies utilisées

- **Streamlit** : Framework pour le dashboard
- **Pandas** : Manipulation des données
- **Plotly** : Visualisations interactives
- **PyTorch** : Framework de deep learning (requis pour TabNet)
- **Pytorch-tabnet** : Implémentation TabNet pour PyTorch
- **Scikit-learn** : Machine learning et preprocessing
- **Joblib** : Sauvegarde et chargement des modèles
- **TA** : Calcul d'indicateurs techniques
- **CCXT** : API pour récupérer les données Binance

## 📝 License

Ce projet fait partie d'une formation OpenClassRooms.

