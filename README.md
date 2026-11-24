# Prévision de l'or (XAU/USD) : séries temporelles

Projet du cours *Advanced Time Series* (ENSAI, 3A, 2026-2027): *Stock Price Prediction*, appliqué au cours de l'or en dollars (XAU/USD).

## Objectif

Modéliser et prévoir la dynamique de l'or à partir de son historique de prix, en comparant plusieurs familles de modèles :

- un **benchmark naïf** (marche aléatoire, moyenne historique) ;
- des modèles de la famille **ARIMA** pour les rendements ;
- des modèles **GARCH** (GARCH, GJR-GARCH, EGARCH) pour la volatilité ;
- éventuellement un modèle de **machine learning** (LSTM ou gradient boosting).

Le prix de l'or étant proche d'une marche aléatoire, l'analyse porte sur les **log-rendements** et sur la **volatilité conditionnelle**. Chaque modèle est évalué face au benchmark naïf, pour juger si la complexité apporte un gain réel.

## Données

| Élément | Choix |
|---|---|
| Actif | XAU/USD (or spot en dollars) ou GC=F (futures or COMEX) |
| Source | Yahoo Finance (`yfinance`) |
| Fréquence | Journalière (Daily) |
| Période | 2005-01-01 → Présent (20+ ans d'historique) |
| Colonnes | Date, Open, High, Low, Close, Adj Close, Volume |

**Remarques**

- Le spot XAU/USD est un marché OTC: il n'existe pas de prix officiel unique, et les sources peuvent différer légèrement.
- Le spot n'a pas de volume réel (seulement du tick volume selon le fournisseur). Le volume n'est pas utilisé, sauf mention contraire.
- Le fichier de données est **figé dans `data/`** pour que tout le groupe travaille sur les mêmes chiffres. Ne pas le retélécharger sans prévenir l'équipe.

## Structure du dépôt

```
xauusd-timeseries/
├── README.md
├── requirements.txt
├── data/                  # CSV figé (brut et nettoyé)
├── scripts/
│   └── download_data.py   # téléchargement du CSV
├── notebooks/
│   ├── 01_eda.ipynb           # exploration, rendements, ACF/PACF, stationnarité
│   ├── 02_arima.ipynb         # ARIMA sur les log-rendements
│   ├── 03_garch.ipynb         # modèles de volatilité
│   ├── 04_ml.ipynb            # modèle ML (optionnel)
│   └── 05_evaluation.ipynb    # comparaison des modèles
└── figures/               # graphiques exportés pour les slides
```

## Installation

Python 3.10 ou plus récent recommandé.

```bash
git clone [URL du dépôt]
cd xauusd-timeseries
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Dépendances principales : `pandas`, `polars`, `numpy`, `statsmodels`, `arch`, `scikit-learn`, `matplotlib`, `yfinance`, `jupyter`.

## Utilisation

1. Télécharger (ou régénérer) les données :
   ```bash
   python scripts/download_data.py
   ```
2. Exécuter les notebooks dans l'ordre (`01` → `05`). Chaque notebook lit le CSV dans `data/` et exporte ses graphiques dans `figures/`.

## Méthodologie

1. **Prétraitement** : dates, valeurs manquantes, outliers, calcul des log-rendements `r_t = ln(P_t / P_{t-1})`.
2. **Analyse exploratoire** : tendance, queues épaisses, clusters de volatilité, ACF/PACF.
3. **Stationnarité** : tests ADF et KPSS sur le prix et sur les rendements.
4. **Modélisation** :
   - benchmark naïf ;
   - ARIMA, ordre choisi par AIC/BIC, diagnostic des résidus (Ljung-Box) ;
   - GARCH, GJR-GARCH, EGARCH sur la volatilité ;
   - modèle ML (optionnel).
5. **Validation** : séparation chronologique (jamais aléatoire) et validation **walk-forward**.
6. **Évaluation** : MAE, MSE, RMSE, précision directionnelle, test de Diebold-Mariano contre le benchmark ; pour la volatilité, comparaison à la volatilité réalisée.
7. **Ouverture** : variables exogènes (indice dollar, taux réels, VIX), VaR/ES et backtesting.

## Résultats

*À compléter une fois les modèles évalués : tableau comparatif des métriques, meilleur modèle, interprétation.*

## Limites

- Les rendements de l'or sont quasi imprévisibles : un gain par rapport au benchmark naïf n'est pas garanti.
- Les performances passées ne garantissent pas les performances futures, et les résultats dépendent de la période choisie.
- Ce projet est académique et ne constitue pas un conseil en investissement.

## Références

- Documents du cours *Advanced Time Series* (ENSAI)
- Documentation : [statsmodels](https://www.statsmodels.org), [arch](https://arch.readthedocs.io)
- [à compléter : articles et sources de données]