import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'datasets.xlsx'
data_cm = pd.read_excel(file_path, sheet_name='CORRECTIVE MAINTANCE')

# Data Exploration
# 1. Check for missing values
missing_values = data_cm.isnull().sum()
print("Missing Values:")
print(missing_values)

# 2. Value counts for key columns: 'Jenis Gangguan', 'Penyebab', 'ZONA', 'SITE', 'SOLUSI', and 'Merk Modem'
print("\nDistribusi Jenis Gangguan:")
print(data_cm['Jenis Gangguan'].value_counts())

print("\nDistribusi Penyebab:")
print(data_cm['Penyebab'].value_counts())

print("\nDistribusi ZONA:")
print(data_cm['ZONA'].value_counts())

print("\nDistribusi SITE:")
print(data_cm['SITE'].value_counts())

print("\nDistribusi SOLUSI:")
print(data_cm['SOLUSI'].value_counts())

print("\nDistribusi Merk Modem:")
print(data_cm['Merk Modem'].value_counts())

# 3. Visualize distributions
# Jenis Gangguan
plt.figure(figsize=(10, 6))
sns.countplot(y='Jenis Gangguan', data=data_cm, order=data_cm['Jenis Gangguan'].value_counts().index)
plt.title('Distribusi Jenis Gangguan')
plt.xlabel('Jumlah')
plt.ylabel('Jenis Gangguan')
plt.show()

# Penyebab
plt.figure(figsize=(10, 6))
sns.countplot(y='Penyebab', data=data_cm, order=data_cm['Penyebab'].value_counts().index[:10])  # Top 10
plt.title('Distribusi Penyebab (Top 10)')
plt.xlabel('Jumlah')
plt.ylabel('Penyebab')
plt.show()

# ZONA
plt.figure(figsize=(10, 6))
sns.countplot(y='ZONA', data=data_cm, order=data_cm['ZONA'].value_counts().index)
plt.title('Distribusi ZONA')
plt.xlabel('Jumlah')
plt.ylabel('ZONA')
plt.show()

# SITE
plt.figure(figsize=(10, 6))
sns.countplot(y='SITE', data=data_cm, order=data_cm['SITE'].value_counts().index[:10])  # Top 10
plt.title('Distribusi SITE (Top 10)')
plt.xlabel('Jumlah')
plt.ylabel('SITE')
plt.show()

# SOLUSI
plt.figure(figsize=(10, 6))
sns.countplot(y='SOLUSI', data=data_cm, order=data_cm['SOLUSI'].value_counts().index[:10])  # Top 10
plt.title('Distribusi SOLUSI (Top 10)')
plt.xlabel('Jumlah')
plt.ylabel('SOLUSI')
plt.show()

# Merk Modem
plt.figure(figsize=(10, 6))
sns.countplot(y='Merk Modem', data=data_cm, order=data_cm['Merk Modem'].value_counts().index[:10])  # Top 10
plt.title('Distribusi Merk Modem (Top 10)')
plt.xlabel('Jumlah')
plt.ylabel('Merk Modem')
plt.show()
