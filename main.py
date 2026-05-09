import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ticker = "^NSEI"
start_date = "2010-01-01"
end_date = "2026-05-08"
datapoints_range = (50,401)
test_start = "2020-01-01"

data = yf.download(ticker, start=start_date, end=end_date)
data.dropna(inplace=True)

df = pd.DataFrame(index=data.index)
df["Close"] = data["Close"]


df["Direction"] = np.where(df["Close"] > df["Close"].shift(1), 1, 0)
df["Direction"] = df["Direction"].shift(-1)

df["Returns"] = df["Close"].pct_change()

fibo = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
for i in fibo:
    df["MA_" + str(i)] = df["Close"].rolling(i).mean()

for i in range(1, len(fibo)):
    j = i-1
    a = fibo[j]
    b = fibo[i]
    lo = df["MA_" + str(a)]
    hi = df["MA_" + str(b)]
    con1 = (lo < hi) & (lo.shift(1) >= hi.shift(1))
    con2 = (lo > hi) & (lo.shift(1) <= hi.shift(1))
    df["MA_" + str(a) + "_" + str(b)] = np.select([con1, con2], [1, 2], 0)
df.dropna(inplace=True)

x = df.loc[:, "MA_2_3":"MA_144_233"]
y = df["Direction"]

scaler = StandardScaler()
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)

acc = pd.DataFrame(index=range(*datapoints_range))
acc["Accuracy"] = np.nan
acc["Cumulative_Returns"] = np.nan
acc["Sharpe_Ratio"] = np.nan

for datapoints in range(*datapoints_range):
    print(datapoints)
    results = pd.DataFrame(index=df.index)
    results["Predicted"] = np.nan
    results["Actual"] = y.loc[results.index]
    for i in df.loc[test_start:].index:
        train_curr_data = x.loc[x.index<i]
        if len(train_curr_data) > datapoints:
            train_curr_data = train_curr_data.iloc[-datapoints:]
        
        train_target = y.loc[train_curr_data.index]
        x_scaled = scaler.fit_transform(train_curr_data)
        model.fit(x_scaled, train_target)

        x_scaled = scaler.transform(x.loc[[i]])
        prob = model.predict(x_scaled)
        results.loc[i, "Predicted"] = prob
    results.dropna(inplace=True)
    accuracy = (results["Actual"] == results["Predicted"]).mean()
    
    results["Returns"] = df["Returns"].shift(-1) * results["Predicted"]
    results["Cumulative_Returns"] = (1+results["Returns"].fillna(0)).cumprod()
    
    acc.loc[datapoints, "Accuracy"] = accuracy * 100
    acc.loc[datapoints, "Sharpe_Ratio"] = (results["Returns"].mean() / results["Returns"].std()) * np.sqrt(252)
    acc.loc[datapoints, "Cumulative_Returns"] = results["Cumulative_Returns"].iloc[-1]
sorted_acc = acc.sort_values(by="Cumulative_Returns", ascending=False)
print(sorted_acc)
sorted_acc.to_csv("Accuracy_by_datapoints_day_closing.csv")