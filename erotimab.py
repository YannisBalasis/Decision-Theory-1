import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
import numpy as np



# Φόρτωση του CSV αρχείου
df = pd.read_csv('final_close_prices.csv', parse_dates=['Date'])

# Ορισμός των χαρακτηριστικών και στόχου
N = 25
feature_columns = [f'close_t-{i}' for i in range(1, N + 1)]
target_column = 'Close'

# Χωρισμός των δεδομένων σε εκπαίδευση και επικύρωση
df['Year'] = df['Date'].dt.year
train_data = df[df['Year'] < 2024]
validation_data = df[df['Year'] == 2024]

X_train = train_data[feature_columns]
y_train = train_data[target_column]
X_val = validation_data[feature_columns]
y_val = validation_data[target_column]

# Κατασκευή πολυωνυμικού μοντέλου με L2  κανονικοποίηση
degree = 2  # Βαθμός 
alpha_l2 = 0.5  # Ρυθμός κανονικοποίησης για L2

ridge_model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha_l2))
ridge_model.fit(X_train, y_train)

# Προβλέψεις και μετρικές για το L2 
y_train_pred_ridge = ridge_model.predict(X_train)
y_val_pred_ridge = ridge_model.predict(X_val)

train_mae_ridge = mean_absolute_error(y_train, y_train_pred_ridge)
train_mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
val_mae_ridge = mean_absolute_error(y_val, y_val_pred_ridge)
val_mse_ridge = mean_squared_error(y_val, y_val_pred_ridge)

print("Ridge Regression (L2) Metrics")
print(f"Training MSE: {train_mse_ridge:.2f}")
print(f"Training MAE: {train_mae_ridge:.2f}")
print(f"Validation MSE: {val_mse_ridge:.2f}")
print(f"Validation MAE: {val_mae_ridge:.2f}")

# Κατασκευή πολυωνυμικού μοντέλου με L1  κανονικοποίηση
alpha_l1 = 0.01  # Ρυθμός κανονικοποίησης για L1

lasso_model = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha_l1, max_iter=10000))
lasso_model.fit(X_train, y_train)

# Προβλέψεις και μετρικές για το L1 
y_train_pred_lasso = lasso_model.predict(X_train)
y_val_pred_lasso = lasso_model.predict(X_val)

train_mae_lasso = mean_absolute_error(y_train, y_train_pred_lasso)
train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
val_mae_lasso = mean_absolute_error(y_val, y_val_pred_lasso)
val_mse_lasso = mean_squared_error(y_val, y_val_pred_lasso)

print("\nLasso Regression (L1) Metrics")
print(f"Training MSE: {train_mse_lasso:.2f}")
print(f"Training MAE: {train_mae_lasso:.2f}")
print(f"Validation MSE: {val_mse_lasso:.2f}")
print(f"Validation MAE: {val_mae_lasso:.2f}")

# Εκτύπωση παραμέτρων των μοντέλων
print("\nRidge Model Coefficients:")
ridge_coefs = ridge_model.named_steps['ridge'].coef_
for i, coef in enumerate(ridge_coefs):
    print(f"Coefficient {i}: {coef:.4f}")
print(f"Intercept: {ridge_model.named_steps['ridge'].intercept_:.4f}")

print("\nLasso Model Coefficients:")
lasso_coefs = lasso_model.named_steps['lasso'].coef_
for i, coef in enumerate(lasso_coefs):
    print(f"Coefficient {i}: {coef:.4f}")
print(f"Intercept: {lasso_model.named_steps['lasso'].intercept_:.4f}")

# Δημιουργία του πολυωνυμικού χαρακτηριστικού 
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# Εκπαίδευση μοντέλου πολυωνυμικής γραμμικής παλινδρόμησης με κανονικοποίηση L2
ridge_model = Ridge(alpha_l2)  
ridge_model.fit(X_train_poly, y_train)

# Προβλέψεις στο validation set
y_val_pred = ridge_model.predict(X_val_poly)

# Υπολογισμός της απόδοσης (mse)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Mean Squared Error Validation Set: {mse:.4f}")

# Προβλέψεις
print(f"Predicts Validation Set: {y_val_pred[:10]}")  
