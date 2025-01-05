import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Φόρτωση του CSV μετά από Gaussian
df = pd.read_csv('final_close_prices.csv', parse_dates=['Date'])


N = 25 #χαρακτηριστικά
feature_columns = [f'close_t-{i}' for i in range(1, N + 1)]
target_column = 'Close'

# Χωρισμός σε Training Set (έως το 2023) και σε Validation Set (2024)
df['Year'] = df['Date'].dt.year
train_data = df[df['Year'] < 2024]
validation_data = df[df['Year'] == 2024]

# Διαχωρισμός χαρακτηριστικών και στόχων για εκπαίδευση και επικύρωση
X_train = train_data[feature_columns]
y_train = train_data[target_column]
X_val = validation_data[feature_columns]
y_val = validation_data[target_column]

#γραμμικο μοντέλο παλινδρόμησης
model = LinearRegression()
model.fit(X_train, y_train)

# Προβλέψεις για το σύνολο εκπαίδευσης και επικύρωσης
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Υπολογισμός μετρικών σφάλματος
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

# Εκτύπωση αποτελεσμάτων
print("Metrics for Training")
print(f" Mean Squared Error : {train_mse:.2f}")
print(f" Mean Absolute Error: {train_mae:.2f}")


print("\nMetrics for Validation")
print(f" Mean Squared Error : {val_mse:.2f}")
print(f" Mean Absolute Error: {val_mae:.2f}")


# Εκτύπωση των παραμέτρων του μοντέλου 
print("\nModel Prameters:")
for i, coef in enumerate(model.coef_):
    print(f"{feature_columns[i]}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Πρόβλεψη της τιμής κλεισίματος της επόμενης ημέρας
latest_data = df[feature_columns].tail(1).values  
next_day_prediction = model.predict(latest_data)

print(f"\nForecast for tomorrow's closing price: {next_day_prediction[0]:.2f}")


