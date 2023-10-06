import sys
from flask import Flask, request
from joblib import load
from utils import f, clean_DF, calculate_recency
import pandas as pd
import numpy as np
import datetime

app = Flask(__name__)

@app.route("/predict_churn", methods=["POST"])
def predict():
    # catching json:
    json_ = request.get_json()
    # convert to DF:
    df = pd.DataFrame(json_)
    # cleaning:
    df = clean_DF(df)    
    # Calculanto agregados por dia:
    bet_by_day = (
      df
      .groupby(["Username", "registration_dt", "ftd_value","age", "date"])
      .agg(
        n_bets=("turnover", lambda x: np.sum(x>0)), 
        turnover=("turnover", np.sum),  
        ggr=("ggr", np.sum)
        )
      .reset_index()
    )
    bet_by_day
    # selecionando variáveis:
    bet_by_day = bet_by_day[["registration_dt", "date", "Username", "ftd_value", "age", "n_bets", "turnover", "ggr"]]
    bet_by_day = bet_by_day.sort_values(by=["registration_dt", "Username"])
    # numero máximo de dias de atividade na plataforma:
    max_days_sample=(bet_by_day["date"].max() - bet_by_day["registration_dt"].min()).days + 1
    # Criando as datas correntes para todos os usuários
    bet_by_day = (
      bet_by_day
      .groupby("Username")
      .apply(f,min_days=max_days_sample)
      .reset_index(drop=True)
    )
    # Filtrando para a data máxima de apostas da base:
    dt_max_bet = df["date"].max()
    bet_by_day = bet_by_day.query("date <= @dt_max_bet")
    ## Calculate Recency:
    
    # recency:
    recency_df = (
      bet_by_day
      .groupby("Username")
      .apply(
        calculate_recency, 
        date_max=datetime.datetime.today()
      )
    )
    recency_df = recency_df.reset_index()
    recency_df.columns = ["Username", "recency_value"]
    ## Calculate Frequency:
    frequency_df = bet_by_day.groupby("Username")["n_bets"].sum()
    frequency_df = frequency_df.reset_index()
    frequency_df.columns = ["Username", "frequency_value"]
    ## Calculate Monetary value:
    revenue_df = bet_by_day.groupby("Username")["ggr"].sum()
    revenue_df = revenue_df.reset_index()
    revenue_df.columns = ["Username", "revenue_value"]
    # turnover value:
    turnover_df = bet_by_day.groupby("Username")["turnover"].sum()
    turnover_df = turnover_df.reset_index()
    turnover_df.columns = ["Username", "turnover_value"]
    # merging:
    rfm_df = recency_df.merge(frequency_df, on="Username")
    rfm_df = rfm_df.merge(revenue_df, on="Username")
    rfm_df = rfm_df.merge(turnover_df, on="Username")
    # calculando ticket medio
    rfm_df["ticket_medio"] = rfm_df["turnover_value"] / rfm_df["frequency_value"]
    # fill na with 0:
    rfm_df["ticket_medio"] = rfm_df["ticket_medio"].fillna(0)
    # adding info about the player:
    rfm_df = rfm_df.merge(
        bet_by_day[["Username", "age", "ftd_value"]].drop_duplicates(),
        on="Username"
    )    
    ## predicting:
    vars_ = [
      'frequency_value', 'revenue_value', 
      'turnover_value', 'ticket_medio', 'age', 'ftd_value'
    ]
    # 1st model: classifier 
    # predicting if LTV = 0
    rfm_df["churn_proba"] = clf.predict_proba(rfm_df[vars_])[:, 1]
    rfm_df["churn_proba"] = np.where(rfm_df["churn_proba"] == 1, .999, rfm_df["churn_proba"])
    rfm_df["churn_proba"] = rfm_df["churn_proba"].round(3)
    
    # returning:
    return rfm_df[["Username", "churn_proba"]].to_json(orient="records")


if __name__ == '__main__':
    # If you don't provide any port the port will be set to 500
    try:
        port = int(sys.argv[1])
    except:
        port = 1234    
    # loading model:
    clf = load("churn_model.joblib")   
    # running debug mode:
    app.run(port=port, debug=True)

