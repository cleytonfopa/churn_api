import sys
from flask import Flask, request
from joblib import load
from utils import calculate_recency
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
    df["data"] = pd.to_datetime(df["data"])
    df["Deposit_Date"] = pd.to_datetime(df["Deposit_Date"])
    # Calculate Recency:
    recency_df = (
      df
      .groupby("Username")
      .apply(
        calculate_recency, 
        date_max=datetime.datetime.today()
      )
    )
    recency_df = recency_df.reset_index()
    recency_df.columns = ["Username", "recency_value"]
    ## Calculate Frequency:
    frequency_df = df.groupby("Username")["n_bets"].sum()
    frequency_df = frequency_df.reset_index()
    frequency_df.columns = ["Username", "frequency_value"]
    ## Calculate Monetary value:
    monetary_df = df.groupby("Username")["ggr"].sum()
    monetary_df = monetary_df.reset_index()
    monetary_df.columns = ["Username", "monetary_value"]
    # Calculate turnover value:
    turnover_df = df.groupby("Username")["turnover"].sum()
    turnover_df = turnover_df.reset_index()
    turnover_df.columns = ["Username", "turnover_value"]
    # merging:
    rfm_df = recency_df.merge(frequency_df, on="Username")
    rfm_df = rfm_df.merge(monetary_df, on="Username")
    rfm_df = rfm_df.merge(turnover_df, on="Username")
    # calculando ticket medio
    rfm_df["ticket_medio"] = rfm_df["turnover_value"] / rfm_df["frequency_value"]
    # fill na with 0:
    rfm_df["ticket_medio"] = rfm_df["ticket_medio"].fillna(0)
    # adding info about the player:
    rfm_df = rfm_df.merge(
        df[["Username", "age", "ftd_value"]].drop_duplicates(),
        on="Username"
    )
    ## predicting:
    vars_ = [
      'recency_value', 'frequency_value', 'monetary_value', 'turnover_value',
      'ticket_medio', 'ftd_value', 'age'
    ]
    # predicting churn
    rfm_df["churn_proba"] = clf.predict_proba(rfm_df[vars_])[:, 1]
    rfm_df["churn_proba"] = np.where(
      rfm_df["churn_proba"] == 1, 
      .999, 
      rfm_df["churn_proba"]
    )
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

