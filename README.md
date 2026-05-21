# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Get the independent variable X and dependent variable Y.

2.Calculate the mean of the X -values and the mean of the Y -values.

3.Find the slope m of the line of best fit using the formula.

4.Compute the y -intercept of the line by using the formula.

5.Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Syed Shamsheer Ali
RegisterNumber:  212225220114
*/
import pandas as pd# Load Dataset
dataset = pd.read_csv("C:/Users/acer/Downloads/Salary.csv")

print(dataset.head())

# Correct Feature Selection
X = dataset.iloc[:, 1:2].values   # Level
y = dataset.iloc[:, 2].values     # Salary

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
import numpy as np
import matplotlib.pyplot as plt

X_grid = np.arange(X.min(), X.max(), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y)
plt.plot(X_grid, regressor.predict(X_grid))
plt.title("Decision Tree Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
```


## Output:
<img width="730" height="685" alt="image" src="https://github.com/user-attachments/assets/4a8a1950-0e2c-4806-a9f2-35b8044834ff" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
