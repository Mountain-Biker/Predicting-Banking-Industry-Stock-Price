######## code for data mining intermediate report
import pandas as pd
import sklearn
import numpy as np
import datetime
import statsmodels.api as sm

#### preprocess data
## bank index 
bank_index_df = pd.read_csv('KBW_bank_index.csv')
bank_index_adj_close = bank_index_df.loc[:, "Adj Close"]
bank_index_monthly_change = np.empty(len(bank_index_adj_close),)
bank_index_monthly_change[:] = np.nan

for i in range(0, len(bank_index_monthly_change) -1):
    bank_index_monthly_change[i] = (bank_index_adj_close[i+1]- bank_index_adj_close[i]) / bank_index_adj_close[i]

bank_index_df['monthly_change'] = bank_index_monthly_change

# change date format to "yyyy-mm-dd"
date = bank_index_df.loc[:, "Date"]
new_date = []
for i in range(0,len(date)):
    new_date.append(datetime.datetime.strptime(date[i], "%m/%d/%y").strftime("20%y-%m-%d"))

bank_index_df['Date'] = new_date

# get the data between date 2011-01-01 and 2020-01-01
date_start = '2011-01-01'
date_end = '2019-12-01'
row_start = bank_index_df[bank_index_df['Date'] == date_start].index[0]
row_end = bank_index_df[bank_index_df['Date'] == date_end].index[0]
bank_mom_change_201101_201912 = bank_index_monthly_change[row_start:row_end]
# print(bank_index_monthly_change_201101_201912)


## commercial and industry loans
comm_indus_loans_df = pd.read_excel('commercial_and_industry_loans.xls', sheet_name='Sheet1')

loans = comm_indus_loans_df.loc[:,"BUSLOANS"]
loans_yoy_change = np.empty(len(loans),)
loans_yoy_change[:] = np.nan
for i in range(0,len(loans)):
    if (i >= 0 and i-12 >= 0):
        loans_yoy_change[i] = (loans[i]-loans[i-12]) / loans[i-12]

comm_indus_loans_df["loans_yoy_change"] = loans_yoy_change
# print(comm_indus_loans_df)

# get the data between date 2011-01-01 and 2020-01-01
date_start = '2011-01-01'
date_end = '2019-12-01'
comm_indus_loans_df = comm_indus_loans_df.rename(columns={'observation_date': 'Date'})
row_start = comm_indus_loans_df[comm_indus_loans_df['Date'] == date_start].index[0]
row_end = comm_indus_loans_df[comm_indus_loans_df['Date'] == date_end].index[0]
loans_yoy_change_201101_201912 = loans_yoy_change[row_start:row_end]
# print(loans_yoy_change_201101_201912)



## PPI
ppi_df = pd.read_excel('PPI.xlsx', sheet_name="WPSFD4")
# print(ppi_df)
ppi = ppi_df.loc[:, "Observation Value"]
ppi_yoy_change = np.empty(len(ppi,))
ppi_yoy_change[:]= np.nan
for i in range(0, len(ppi)):
    if (i>=0 and i-12>=0):
        ppi_yoy_change[i] = (ppi[i] - ppi[i-12])/ ppi[i-12]

ppi_df["ppi_yoy_change"] = ppi_yoy_change
# print(ppi_df)

# get the data between date 2011-01-01 and 2020-01-01
date_start = '2011-01-01'
date_end = '2019-12-01'
row_start = ppi_df[ppi_df['Date'] == date_start].index[0]
row_end = ppi_df[ppi_df['Date'] == date_end].index[0]
ppi_yoy_change_201101_201912 = ppi_yoy_change[row_start:row_end]
# print(ppi_yoy_change_201101_201912)



## Purchasing Manager Index
pmi_df = pd.read_excel('Purchasing_Manager_Index.xlsx', sheet_name='Sheet1')

pmi_actual = pmi_df.loc[:,"Actual"]
pmi_forecast = pmi_df.loc[:, "Forecast"]
pmi_forecast_diff = pmi_actual - pmi_forecast
pmi_df['forecast_diff'] = pmi_forecast_diff
# print(pmi_df)

# get the data between date 2011-01-01 and 2020-01-01
date_start = '2011-01-01'
date_end = '2019-12-01'
row_start = pmi_df[pmi_df['Date'] == date_start].index[0]
row_end = pmi_df[pmi_df['Date'] == date_end].index[0]
pmi_actual_201101_201912 = pmi_actual[row_start:row_end]
pmi_forecast_diff_201101_201912 = pmi_forecast_diff[row_start:row_end]
# print(pmi_forecast_diff_201101_201912)


## Unemployment Rate
unemp_df = pd.read_excel('Unemployment_rate.xlsx', sheet_name='Sheet1')
# print(unemp_df)
unemp = unemp_df.loc[:, "Unemployment Rate"]
unemp = unemp/100.0
unemp_df["Unemployment Rate"] = unemp
unemp_monthly_change = np.empty(len(unemp),)
unemp_monthly_change[:] = np.nan
for i in range(1,len(unemp)):
    unemp_monthly_change[i] = unemp[i]-unemp[i-1]

unemp_df['unemp_monthly_change'] = unemp_monthly_change
# print(unemp_df)

# get the data between date 2011-01-01 and 2020-01-01
date_start = '2011-01-01'
date_end = '2019-12-01'
row_start = unemp_df[unemp_df['Date'] == date_start].index[0]
row_end = unemp_df[unemp_df['Date'] == date_end].index[0]
unemp_201101_201912 = unemp[row_start:row_end]
unemp_monthly_change_201101_201912 = unemp_monthly_change[row_start:row_end]
# print(unemp_monthly_change_201101_201912)
