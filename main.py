import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
movies = pd.read_csv('movie_dataset.csv')

# Fill missing directors and budget with 'Unknown' and median respectively
movies['director'].fillna('Unknown', inplace=True)
movies['budget'].fillna(movies['budget'].median(), inplace=True)

# Scale the budget and revenue using StandardScaler
scaler_budget = StandardScaler()
movies['budget_scaled'] = scaler_budget.fit_transform(movies['budget'].values.reshape(-1, 1))

scaler_revenue = StandardScaler()
movies['revenue_scaled'] = scaler_revenue.fit_transform(movies['revenue'].values.reshape(-1, 1))

# One-hot encoding for director
movies_encoded = pd.get_dummies(movies['director'])

# Combine encoded and scaled budget features
movies_final = pd.concat([movies['budget_scaled'], movies_encoded], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(movies_final, movies['revenue_scaled'], test_size=0.2, random_state=42)

# Train a random forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred_scaled = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_scaled)

# Inverse transform the scaled predictions
y_pred = scaler_revenue.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Create a table to display the predictions and actual values
table = pd.DataFrame({'Movie': movies.loc[y_test.index, 'title'],
                      'Prediction': y_pred.flatten().astype(int),
                      'Actual Revenue': movies.loc[y_test.index, 'revenue'].astype(int)})

# Print the table
print(table)
print()

# Print the MSE
print("Mean Squared Error:", mse)
