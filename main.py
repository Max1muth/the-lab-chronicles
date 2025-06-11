import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import openpyxl

df = pd.read_excel("датасет-1.xlsx")
# print(df)
pred = pd.read_excel("prediction_price.xlsx")
# print(pred)

plt.scatter(df.area, df.price, color='red')
plt.xlabel("площадь(м**2)")
plt.ylabel("стоимость(млн.руб)")
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# print(reg.predict([[38]])) # [3.52747127]

price = reg.coef_ * df.area + reg.intercept_

plt.scatter(df.area, df.price, color='red')
plt.plot(df.area, reg.predict(df[['area']]))
plt.xlabel("площадь(м**2)")
plt.ylabel("стоимость(млн.руб)")
plt.show()

p = reg.predict(pred)

pred["predicted_prices"] = p
pred.to_excel("new.xlsx", index=False)

print(pred)
