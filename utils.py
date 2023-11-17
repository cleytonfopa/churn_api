import pandas as pd 
import numpy as np

# função para calcular Recency
def calculate_recency(df, date_max):
    recency_value = date_max - df.query("n_bets > 0")["data"].max()
    # Se naT (nao houve nenhuma aposta):
    if pd.isnull(recency_value):
        recency_value = date_max - df["Deposit_Date"].max()
    recency_value = recency_value.days
    return recency_value




