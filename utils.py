import pandas as pd 
import numpy as np

def clean_DF(df):
    # removing cancelled bets:
    df = df.query("Cancelled == 0")
    # fixing date columns:
    df["Reg_Date"] = pd.to_datetime(df["Reg_Date"])
    df["DOB"] = pd.to_datetime(df["DOB"])
    # date
    df["registration_dt"] = df["Reg_Date"].dt.date
    # idade:
    df["age"] = pd.to_datetime(df["registration_dt"]).dt.year - df["DOB"].dt.year
    # rename
    df.rename(columns={"Amount": "ftd_value"}, inplace=True)
    # columns:
    players_input = ["User_ID", "name", "Username",
                     "registration_dt", "ftd_value",
                     "deposit_dt", "age", "no-affiliate"]
    # fixing numerical values:
    df["Bet"] = df["Bet"].str.replace(",", ".").astype("float")
    df["Win"] = df["Win"].str.replace(",", ".").astype("float")
    df["Cancelled"] = df["Cancelled"].astype("int")
    # date features:
    df["Date_Time"] = pd.to_datetime(df["Date_Time"])
    df["date"] = df["Date_Time"].dt.date
    # ggr
    df["ggr"] = df["Bet"] - df["Win"]
    # renaming
    df.rename(columns={"Bet": "turnover"}, inplace=True)
    return df

# function to guarantee at least 180 days:
def f(x, min_days=181):
    # convert to datetime
    x["date"] = pd.to_datetime(x["date"])
    x["registration_dt"] = pd.to_datetime(x["registration_dt"])
    # merge
    x = pd.merge(
        pd.DataFrame({"date": pd.date_range(x["registration_dt"].min(), periods=min_days)}),
        x,
        on="date",
        how="left"
    )
    # processing:
    if x.dropna().shape[0] == 0:
        x= x.dropna()
    else:    
       x["ftd_value"].fillna(x["ftd_value"].dropna().unique()[0], inplace=True)    
       x["age"].fillna(x["age"].dropna().unique()[0], inplace=True)
       x["Username"].fillna(x["Username"].dropna().unique()[0], inplace=True)
       #x["no-affiliate"].fillna(x["no-affiliate"].dropna().unique()[0], inplace=True)
       x["n_bets"].fillna(0, inplace=True)
       x["turnover"].fillna(0, inplace=True)
       x["ggr"].fillna(0, inplace=True)
       x["registration_dt"].fillna(x["registration_dt"].dropna().unique()[0], inplace=True)
       return x

# função para calcular Recency
def calculate_recency(df, date_max):
    recency_value = date_max - df.query("n_bets > 0")["date"].max()
    # Se naT (nao houve nenhuma aposta):
    if pd.isnull(recency_value):
        recency_value = date_max - df["registration_dt"].max()
    recency_value = recency_value.days
    return recency_value

# helper function to get the quantiles from the trees in the Forest:
def get_quantiles(x, q=[.05, .95]):
   import numpy as np    
   # calculando o q1 e q3
   q1 = np.quantile(x, [.25])[0]
   q3 = np.quantile(x, [.75])[0]
   # calculando iqr:
   iqr = q3 - q1
   # retirando os outliers:
   (x >= (q1 - 1.5*iqr)) & (x <= (q3 + 1.5*iqr))   
   new_x = x[(x >= (q1 - 1.5*iqr)) & (x <= (q3 + 1.5*iqr))]   
   # pegando o lower bound
   lower_ = np.quantile(new_x, [q[0]])[0]
   # calculando a média:
   avg_ = np.mean(new_x)
   # pegando o upper bound:
   upper_ = np.quantile(new_x, [q[1]])[0]
   lower_, avg_, upper_
   return lower_, avg_, upper_

# helper function to make a prediction interval:
def predict_interval(rgr, data, q=[.1, .9]):
    # applying preprocessing:
    newdata = rgr["imputer"].transform(data)
    newdata = rgr["scaling"].transform(data)
    # predicting using each tree in the Forest:
    all_pred = [tree.predict(newdata) for tree in rgr["regressor"].estimators_]
    all_pred = pd.DataFrame(all_pred, columns=data.index)
    all_pred = all_pred.T
    # get the lower, avg and upper:
    pred = all_pred.apply(get_quantiles, axis=1)
    pred = pd.DataFrame(pred.to_list(), columns=["lower", "avg", "upper"], index=data.index)
    return pred




