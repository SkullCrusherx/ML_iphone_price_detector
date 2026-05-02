import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error, r2_score

while True:
    csv = pd.read_csv('new.csv')
    X = csv[['Series','RAM_GB','Storage_GB','Release_Year','Screen_Size','Battery_mAh']].values
    y = csv[['Price']].values

    reg = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2:", r2)
    if r2 >= 0.99 and mse <=200000:
        joblib.dump(reg,'2nd_hand_iphone.pkl')
        print("Model saved")
        break
    else:
        print("not done")

