import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

DATA = "data"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

level_salary = pd.read_csv(os.path.join(DATA, 'Levels_Fyi_Salary_Data.csv'))
cost_of_living = pd.read_csv(os.path.join(DATA, 'cost_of_living.csv'))
country_codes = pd.read_excel(os.path.join(DATA, 'country_codes.xlsx'))
ds_salaries = pd.read_csv(os.path.join(DATA, 'ds_salaries.csv'))

# Cost of living: split "City, Country"
city_country = cost_of_living['City'].str.split(', ', expand=True)
cost_of_living['City'] = city_country[0]
cost_of_living['Country'] = city_country[1]
cost_of_living = cost_of_living.drop(columns=['City','Rank'])

# Average by country
avg_cofl = cost_of_living.groupby('Country').mean().reset_index()

# Map country codes for ds_salaries
code_map = dict(zip(country_codes['Alpha-2 code'], country_codes['Country']))
ds = ds_salaries[['job_title','salary_in_usd','company_location','experience_level']].copy()
ds = ds[ds['job_title'] == 'Data Scientist']
ds['Country'] = ds['company_location'].map(code_map)
ds = ds.dropna(subset=['Country'])

# Remove salary outliers via IQR
Q1, Q3 = ds['salary_in_usd'].quantile(0.25), ds['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
ds_filt = ds[(ds['salary_in_usd'] >= lower) & (ds['salary_in_usd'] <= upper)]

# Boxplots
plt.figure(figsize=(10,4))
sns.boxplot(x=ds['salary_in_usd'], orient='h')
plt.title("Salary (USD) — raw")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "salary_box_raw.png"))

plt.figure(figsize=(10,4))
sns.boxplot(x=ds_filt['salary_in_usd'], orient='h')
plt.title("Salary (USD) — filtered (IQR)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "salary_box_iqr.png"))

# Average salary by country
sal_by_country = ds_filt.groupby('Country')['salary_in_usd'].mean().reset_index()
sal_by_country.rename(columns={'salary_in_usd':'SalaryUSD'}, inplace=True)

# Merge with cost-of-living
merged = sal_by_country.merge(avg_cofl, on='Country', how='left').dropna()

# Ratios
index_cols = ['Cost of Living Index','Rent Index','Cost of Living Plus Rent Index',
              'Groceries Index','Restaurant Price Index','Local Purchasing Power Index']
for col in index_cols:
  merged[f'Salary/{col}'] = merged['SalaryUSD'] / merged[col]

# Plot top-5 for each ratio
for col in index_cols:
  ratio_col = f'Salary/{col}'
  top5 = merged.nlargest(5, ratio_col)
  plt.figure(figsize=(8,4))
  sns.barplot(data=top5, x=ratio_col, y='Country', edgecolor=".5", linewidth=1)
  plt.title(f'Top 5 Countries by {ratio_col}')
  plt.tight_layout()
  plt.savefig(os.path.join(OUT, f'top5_{ratio_col.replace(" ","_").replace("/","_")}.png'))
