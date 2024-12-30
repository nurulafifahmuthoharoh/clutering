import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = 'datasets.xlsx'
data_cm = pd.read_excel(file_path, sheet_name='CORRECTIVE MAINTANCE')

# Data Cleaning
# Remove unnecessary columns
columns_to_drop = ['Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18']
data_cm = data_cm.drop(columns=[col for col in columns_to_drop if col in data_cm.columns])

# Handle missing values
# Drop rows where 'Jenis Gangguan', 'Penyebab', or 'ZONA' are missing
data_cm = data_cm.dropna(subset=['Jenis Gangguan', 'Penyebab', 'ZONA'])

# Fill missing values in 'SITE' and 'Merk Modem' with 'Unknown'
data_cm['SITE'] = data_cm['SITE'].fillna('Unknown')
data_cm['Merk Modem'] = data_cm['Merk Modem'].fillna('Unknown')

# Normalize column values
# Standardize ZONA names (e.g., 'ZONA3' to 'ZONA 3')
data_cm['ZONA'] = data_cm['ZONA'].str.replace('ZONA', '').str.strip()
data_cm['ZONA'] = 'ZONA ' + data_cm['ZONA']

# Data Transformation
# Encode categorical columns ('Jenis Gangguan', 'Penyebab', 'ZONA', 'Merk Modem')
le_gangguan = LabelEncoder()
le_penyebab = LabelEncoder()
le_zona = LabelEncoder()
le_modem = LabelEncoder()

# Fit and transform data
data_cm['Jenis Gangguan'] = le_gangguan.fit_transform(data_cm['Jenis Gangguan'])
data_cm['Penyebab'] = le_penyebab.fit_transform(data_cm['Penyebab'])
data_cm['ZONA'] = le_zona.fit_transform(data_cm['ZONA'])
data_cm['Merk Modem'] = le_modem.fit_transform(data_cm['Merk Modem'])

# Save encoders for deployment
joblib.dump(le_gangguan, 'le_gangguan.pkl')
joblib.dump(le_penyebab, 'le_penyebab.pkl')
joblib.dump(le_zona, 'le_zona.pkl')
joblib.dump(le_modem, 'le_modem.pkl')

# Scale numeric columns if applicable
scaler = StandardScaler()
columns_to_scale = ['Jenis Gangguan', 'Penyebab', 'ZONA', 'Merk Modem']
data_cm[columns_to_scale] = scaler.fit_transform(data_cm[columns_to_scale])

# Save the scaler for deployment
joblib.dump(scaler, 'scaler.pkl')

# Save the prepared data for further analysis
data_cm.to_csv('prepared_data.csv', index=False)
print("Data preparation completed. Prepared data saved to 'prepared_data.csv'.")

# Preview prepared data
print("Prepared Data Preview:")
print(data_cm.head())
