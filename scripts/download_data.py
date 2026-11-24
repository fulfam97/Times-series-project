"""
Script de téléchargement des données historiques pour le projet de séries temporelles sur l'or.

Usage :
    python scripts/download_data.py
    python scripts/download_data.py --ticker GC=F --start 2005-01-01
    python scripts/download_data.py --ticker XAUUSD=X --with-covariates
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Erreur: le package 'yfinance' n'est pas installé.")
    print("Installez-le avec : pip install yfinance")
    sys.exit(1)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Tickers par défaut
DEFAULT_TICKER = "GC=F"  # Futures Or (COMEX) - historique long et propre
SPOT_TICKER = "XAUUSD=X" # Spot Or / USD

# Variables macroéconomiques exogènes couramment utilisées avec l'or
COVARIATE_TICKERS = {
    "DXY": "DX-Y.NYB",   # Indice Dollar US (corrélation inverse historique)
    "VIX": "^VIX",       # Indice de volatilité S&P 500 (mesure du stress/risk-off)
    "TNX_10Y": "^TNX",   # Taux 10 ans US (coût d'opportunité de détenir de l'or)
    "OIL": "CL=F",       # Pétrole WTI (proxy de pressions inflationnistes)
    "SP500": "^GSPC",    # Indice boursier large S&P 500
}


def download_gold_data(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    """Télécharge les données OHLCV de l'or via Yahoo Finance."""
    print(f"--> Téléchargement des cours pour '{ticker}' depuis {start}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"Aucune donnée retournée pour le ticker {ticker}.")

    # Si yfinance retourne un MultiIndex dans les colonnes (version récente)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Nettoyage de l'index
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    # Vérification des colonnes attendues
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_cols = [c for c in expected_cols if c in df.columns]
    df = df[available_cols].copy()

    # Nettoyage des valeurs manquantes éventuelles
    missing_count = df["Close"].isna().sum()
    if missing_count > 0:
        print(f"    Attention : {missing_count} valeurs manquantes détectées dans 'Close' (remplissage forward-fill).")
        df["Close"] = df["Close"].ffill().bfill()

    return df


def download_covariates(start: str, end: str = None) -> pd.DataFrame:
    """Télécharge les variables macroéconomiques et les aligne sur le prix de clôture."""
    print("--> Téléchargement des variables exogènes (DXY, VIX, TNX, OIL, SP500)...")
    cov_dfs = {}
    for name, sym in COVARIATE_TICKERS.items():
        try:
            data = yf.download(sym, start=start, end=end, progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                series = data["Close"]
                series.index = pd.to_datetime(series.index)
                if series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                cov_dfs[name] = series
            else:
                print(f"    Avertissement : données vides pour {name} ({sym})")
        except Exception as e:
            print(f"    Échec du téléchargement pour {name} ({sym}) : {e}")

    df_cov = pd.DataFrame(cov_dfs)
    df_cov.index.name = "Date"
    return df_cov


def main():
    parser = argparse.ArgumentParser(description="Téléchargement des données de l'or pour le projet Time Series.")
    parser.add_argument(
        "--ticker",
        type=str,
        default=DEFAULT_TICKER,
        help=f"Ticker Yahoo Finance de l'actif principal (ex: {DEFAULT_TICKER} pour Futures, {SPOT_TICKER} pour Spot). Défaut: {DEFAULT_TICKER}",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2005-01-01",
        help="Date de début (YYYY-MM-DD). Défaut: 2005-01-01",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Date de fin (YYYY-MM-DD). Défaut: aujourd'hui",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin de sortie du fichier CSV principal (défaut: data/gold_daily.csv)",
    )
    parser.add_argument(
        "--with-covariates",
        action="store_true",
        help="Télécharge également les variables macroéconomiques exogènes dans data/covariates_daily.csv",
    )

    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else (DATA_DIR / "gold_daily.csv")

    try:
        df_gold = download_gold_data(ticker=args.ticker, start=args.start, end=args.end)
        df_gold.to_csv(output_path)
        print(f"\n[OK] Données principales enregistrées dans : {output_path}")
        print(f"     Période : du {df_gold.index.min().strftime('%Y-%m-%d')} au {df_gold.index.max().strftime('%Y-%m-%d')}")
        print(f"     Nombre d'observations : {len(df_gold)}")
        print(f"     Colonnes : {list(df_gold.columns)}")
        print(f"     Dernier cours de clôture : {df_gold['Close'].iloc[-1]:.2f} USD")

        if args.with_covariates:
            df_cov = download_covariates(start=args.start, end=args.end)
            cov_output = DATA_DIR / "covariates_daily.csv"
            df_cov.to_csv(cov_output)
            print(f"\n[OK] Variables exogènes enregistrées dans : {cov_output}")
            print(f"     Variables : {list(df_cov.columns)}")

        print("\nPrêt pour l'exploration dans les notebooks !")

    except Exception as e:
        print(f"\n[Erreur] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
