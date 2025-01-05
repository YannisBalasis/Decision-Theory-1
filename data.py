import requests
import csv
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# Ρυθμίσεις API και συλλοφή δεδομένων
API_KEY = 'NS5KBZ22WFBLK34K'  
STOCK_SYMBOL = 'MCHP'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={STOCK_SYMBOL}&outputsize=full&apikey={API_KEY}'
r = requests.get(url)
data = r.json()

# Εξαγωγή δεδομένων τιμών κλεισίματος από το JSON
time_series = data.get("Time Series (Daily)", {})
csv_data = [("Date", "Close")]  # Κεφαλίδες για το CSV

for date, daily_data in time_series.items():
    close_price = daily_data.get("4. close", None)
    if close_price:
        # Αποθήκευση της ημερομηνίας και τιμής κλεισίματος
        csv_data.append((date, float(close_price)))  

# Αποθήκευση σε CSV αρχείο
csv_file = 'close_prices.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

# Μήνυμα επιβεβαίωσης
print(f"Data saved to {csv_file}")

# Φόρτωση των δεδομένων 
df = pd.read_csv(csv_file, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)  # Ταξινόμηση με βάση ημερομηνίες

# Προσθήκη καθυστερημένων τιμών ως χαρακτηριστικών
N = 50  #  καθυστερημένες τιμες
for i in range(1, N + 1):
    df[f'close_t-{i}'] = df['Close'].shift(i)

# Κενές Γραμμές
df = df.dropna().reset_index(drop=True)

# Αποθήκευση του DataFrame με τα νέα χαρακτηριστικά σε νέο CSV αρχείο
csv_file_with_features = 'close_prices_with_features.csv'
df.to_csv(csv_file_with_features, index=False)
print(f"Data with lag features saved to {csv_file_with_features}")

# Ορισμός σ για Gaussian 
sigma = 2

# Εφαρμογή του φίλτρου λειανσης 
df['Final_Close'] = gaussian_filter1d(df['Close'], sigma=sigma)

# Αποθήκευση του αποτελέσματος σε νέο CSV
filtered_csv_file = 'final_close_prices.csv'
df.to_csv(filtered_csv_file, index=False)