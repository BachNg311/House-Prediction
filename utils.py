import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def getData():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    col = ["LotArea", "LotFrontage", "TotRmsAbvGrd", "SalePrice"]
    train = train[col]
    train = train.dropna()
    X_train, y_train = train[["LotArea", "LotFrontage", "TotRmsAbvGrd"]], train["SalePrice"]
    
    X_test = test[["LotArea", "LotFrontage", "TotRmsAbvGrd"]]
    # print(X_train.shape)

    return X_train, y_train, X_test


def build_model():
    X_train, y_train, X_test = getData()
    X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    print(np.sqrt(mean_squared_error(y_val, pred)))
    print(model.coef_, model.intercept_)
    return model.coef_, model.intercept_

build_model()
