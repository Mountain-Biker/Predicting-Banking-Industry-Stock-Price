#### code for data mining intermediate report
import pandas as pd
import sklearn
import numpy as np
import datetime
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
import datetime

#### preprocess data 

# the date interval of stock price change rate we want to investigate
date_start = '2011-01-01'
date_end = '2019-12-01'

# function date_shift_month is used to get the date after shift delta_month
def date_shift_month(date, delta_month):
    date_shift = pd.to_datetime(date)+pd.DateOffset(months=delta_month)
    return str(date_shift.date())

# index_yoy_change_func is used to calculate the yoy change rate of a economic index
def index_yoy_change_func(index):
    index_yoy_change = np.empty(len(index))
    index_yoy_change[:] = np.nan
    for i in range(12, len(index)):
        index_yoy_change[i] = (index[i]-index[i-12])/index[i-12]

    return index_yoy_change

# index_yoy_change_func is used to calculate the mom change rate of a economic index
def index_mom_change_func(index):
    index_mom_change = np.empty(len(index))
    index_mom_change[:] = np.nan
    for i in range(1, len(index)):
        index_mom_change[i] = (index[i]-index[i-1])/index[i-1]
    
    return index_mom_change


# get the row index of the row which the value of Date equals date
def row_index_func(index_dataframe, date):
    return index_dataframe[index_dataframe['Date'] == date].index[0]


# get_interval_data_func() get the data for a specific time interval from date_start to date_end
# data_name should be a string
def get_interval_data(index_dataframe, data_name, date_start, date_end):
    row_start = row_index_func(index_dataframe, date_start)
    row_end = row_index_func(index_dataframe, date_end)
    return np.array(index_dataframe[data_name][row_start:row_end])


# calculate the Mean squared error of a regression result
def mean_squared_error(y_real, y_esti):
    squared_error = 0
    for i in range(0, len(y_esti)):
        squared_error += (y_real[i] - y_esti[i])**2

    return squared_error/len(y_esti)


# calculate the R square of the regression result
def r_square_func(y_real, y_esti):
    sum_squared_residuals = 0
    total_sum_squares = 0
    y_mean = np.mean(y_real)

    for i in range(0,len(y_real)):
        sum_squared_residuals += (y_real[i] - y_esti[i])**2
        total_sum_squares += (y_real[i] - y_mean)**2

    return 1-(sum_squared_residuals / total_sum_squares)



## explained variable: bank index 
bank_index_df = pd.read_csv('KBW_bank_index.csv')
bank_index_adj_close = bank_index_df.loc[:, "Adj Close"]
bank_index_monthly_change = np.empty(len(bank_index_adj_close),)
bank_index_monthly_change[:] = np.nan

for i in range(1, len(bank_index_monthly_change)):
    bank_index_monthly_change[i] = (bank_index_adj_close[i]- bank_index_adj_close[i-1]) / bank_index_adj_close[i-1]

bank_index_df['monthly_change'] = bank_index_monthly_change

# change date format to "yyyy-mm-dd"
date = bank_index_df.loc[:, "Date"]
new_date = []
for i in range(0,len(date)):
    new_date.append(datetime.datetime.strptime(date[i], "%m/%d/%y").strftime("20%y-%m-%d"))

bank_index_df['Date'] = new_date

# get the data between date interval
row_start = bank_index_df[bank_index_df['Date'] == date_start].index[0]
row_end = bank_index_df[bank_index_df['Date'] == date_end].index[0]
bank_mom_change_interval = bank_index_monthly_change[row_start:row_end]
# print(bank_index_monthly_change_201101_201912)

print(bank_index_df)

#### explaining variables 
##  commercial and industry loans
comm_indus_loans_df = pd.read_excel('commercial_and_industry_loans.xls', sheet_name='Sheet1')

loans = comm_indus_loans_df.loc[:,"BUSLOANS"]
loans_yoy_change = np.empty(len(loans),)
loans_yoy_change[:] = np.nan
for i in range(0,len(loans)):
    if (i >= 0 and i-12 >= 0):
        loans_yoy_change[i] = (loans[i]-loans[i-12]) / loans[i-12]

comm_indus_loans_df["loans_yoy_change"] = loans_yoy_change
# print(comm_indus_loans_df)

# get the data between date interval
date_t0 = date_start # lag=0
date_t1 = date_end
comm_indus_loans_df = comm_indus_loans_df.rename(columns={'observation_date': 'Date'})
row_start = comm_indus_loans_df[comm_indus_loans_df['Date'] == date_t0].index[0]
row_end = comm_indus_loans_df[comm_indus_loans_df['Date'] == date_t1].index[0]
loans_yoy_change_interval = loans_yoy_change[row_start:row_end]
# print(loans_yoy_change_201101_201912)


## PPI
ppi_df = pd.read_excel('PPI.xls', sheet_name="Sheet1")
# print(ppi_df)
ppi = ppi_df.loc[:, "PPI"]
ppi_yoy_change = np.empty(len(ppi,))
ppi_yoy_change[:]= np.nan
for i in range(0, len(ppi)):
    if (i>=0 and i-12>=0):
        ppi_yoy_change[i] = (ppi[i] - ppi[i-12])/ ppi[i-12]

ppi_df["ppi_yoy_change"] = ppi_yoy_change
# print(ppi_df)

# get the data between date 2011-01-01 and 2020-01-01
lag = -1
date_t0 = date_shift_month(date=date_start, delta_month=lag)
date_t1 = date_shift_month(date=date_end, delta_month=lag)
row_start = ppi_df[ppi_df['Date'] == date_t0].index[0]
row_end = ppi_df[ppi_df['Date'] == date_t1].index[0]
ppi_yoy_change_interval = ppi_yoy_change[row_start:row_end]
# print(ppi_yoy_change_201101_201912)



## Purchasing Manager Index
pmi_df = pd.read_excel('Purchasing_Manager_Index.xlsx', sheet_name='Sheet1')

pmi_actual = pmi_df.loc[:,"Actual"]
pmi_forecast = pmi_df.loc[:, "Forecast"]
pmi_forecast_diff = pmi_actual - pmi_forecast
pmi_df['forecast_diff'] = pmi_forecast_diff
# print(pmi_df)

# get the data between date 2011-01-01 and 2020-01-01
lag = -1
date_t0 = date_shift_month(date=date_start, delta_month=lag)
date_t1 = date_shift_month(date=date_end, delta_month=lag)
row_start = pmi_df[pmi_df['Date'] == date_t0].index[0]
row_end = pmi_df[pmi_df['Date'] == date_t1].index[0]
pmi_actual_interval = pmi_actual[row_start:row_end]
pmi_forecast_diff_interval = pmi_forecast_diff[row_start:row_end]
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

# get the data between date interval
lag = -1
date_t0 = date_shift_month(date=date_start, delta_month=lag)
date_t1 = date_shift_month(date=date_end, delta_month=lag)
row_start = unemp_df[unemp_df['Date'] == date_t0].index[0]
row_end = unemp_df[unemp_df['Date'] == date_t1].index[0]
unemp_interval = unemp[row_start:row_end]
unemp_monthly_change_interval = unemp_monthly_change[row_start:row_end]
# print(unemp_monthly_change_201101_201912)


## Personal Income
pi_df = pd.read_excel('PI.xls', sheet_name='Sheet1')
pi = pi_df.loc[:, "PI"]
pi_yoy_change = index_yoy_change_func(pi)
pi_df['pi_yoy_change'] = pi_yoy_change

# get the data between date interval
lag = -2
date_t0 = date_shift_month(date=date_start, delta_month=lag)
date_t1 = date_shift_month(date=date_end, delta_month=lag)
pi_yoy_change_interval = get_interval_data(pi_df, 'pi_yoy_change', date_t0, date_t1)

## PCE price index
pcepi_df = pd.read_excel('PCEPI.xls', sheet_name='Sheet1')
pcepi = pcepi_df.loc[:, "PCEPI"]
pcepi_yoy_change = index_yoy_change_func(pcepi)
pcepi_df['pcepi_yoy_change'] = pcepi_yoy_change

# get the data between date interval
lag = -2
date_t0 = date_shift_month(date=date_start, delta_month=lag)
date_t1 = date_shift_month(date=date_end, delta_month=lag)
pcepi_yoy_change_interval = get_interval_data(pcepi_df, 'pcepi_yoy_change', date_t0, date_t1)


## Durable goods orders 
dgo_df = pd.read_excel('DGORDER.xls', sheet_name='Sheet1')
dgo = dgo_df.loc[:, 'DGORDER']
dgo_mom_change = index_mom_change_func(dgo)
dgo_df['dgo_mom_change'] = dgo_mom_change

# get the data between date interval
lag = -2
date_t0 = date_shift_month(date=date_start, delta_month=lag)
date_t1 = date_shift_month(date=date_end, delta_month=lag)
dgo_mom_change_interval = get_interval_data(dgo_df, 'dgo_mom_change', date_t0, date_t1)


## new housing units started
houst_df = pd.read_excel('HOUST.xls', sheet_name='Sheet1')
houst = houst_df.loc[:,'HOUST']
houst_yoy_change = index_yoy_change_func(houst)
houst_df['houst_yoy_change'] = houst_yoy_change

# get the data between date interval
lag = -1
date_t0 = date_shift_month(date=date_start, delta_month=lag)
date_t1 = date_shift_month(date=date_end, delta_month=lag)
houst_yoy_change_interval = get_interval_data(houst_df, 'houst_yoy_change', date_t0, date_t1)



#### run regression 
## The first linear regression with 6 variables
X = [loans_yoy_change_interval, ppi_yoy_change_interval, pmi_actual_interval,
      pmi_forecast_diff_interval, unemp_interval, unemp_monthly_change_interval]

X = np.array(X)
X = X.transpose()

# regression with scikit learn LinrearRegression()
# reg = LinearRegression().fit(X, bank_mom_change_interval)
# print("reg score:" + str(reg.score(X,bank_mom_change_interval)))
# print("coeff: " +str(reg.intercept_) + " " + str(reg.coef_))

X = sm.add_constant(X)
model = sm.OLS(bank_mom_change_interval,X)

results = model.fit()
print(results.summary())


## The second linear regression with removing PMI and add 4 new variables
X = [loans_yoy_change_interval, ppi_yoy_change_interval, 
       unemp_interval, unemp_monthly_change_interval, 
      pi_yoy_change_interval, pcepi_yoy_change_interval, dgo_mom_change_interval, houst_yoy_change_interval]

X = np.array(X)
X = X.transpose()
X = sm.add_constant(X)
model = sm.OLS(bank_mom_change_interval,X)

results = model.fit()
print(results.summary())




## The third regression model: add polynomial term with degree 2
poly = PolynomialFeatures(2)
X = [loans_yoy_change_interval, ppi_yoy_change_interval, 
       unemp_interval, unemp_monthly_change_interval, dgo_mom_change_interval,
      pi_yoy_change_interval, pcepi_yoy_change_interval]

X = np.array(X)
X = X.transpose()

X_poly_train = poly.fit_transform(X)
X_poly_train = sm.add_constant(X_poly_train)
model = sm.OLS(bank_mom_change_interval,X_poly_train)

results = model.fit()
print(results.summary())


# month = pd.date_range('2011-02-01','2019-12-01', 
#               freq='Y').strftime("%Y-%m").tolist()
# print(month)

# plot the real value and esimated value of training set
y_esti_train = results.predict(X_poly_train)

figure1 = plt.figure()
time = range(0,len(y_esti_train))
plt.plot(time,bank_mom_change_interval,c="r",label='real stock returns of training set')
plt.plot(time,y_esti_train,c="b",label='estimated stock returns')
plt.xlabel("month index",fontsize=12)
plt.ylabel('stock returns',fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=12)
# plt.show()

# test set
test_date_start = '2021-01-01'
test_date_end = '2022-12-01'

bank_mom_change_interval_test = get_interval_data(bank_index_df, 'monthly_change', test_date_start, test_date_end)

# explanatory variables
# X = [loans_yoy_change_interval, ppi_yoy_change_interval, 
#        unemp_interval, unemp_monthly_change_interval, dgo_mom_change_interval,
#       pi_yoy_change_interval, pcepi_yoy_change_interval]
#loan
lag = 0
date_t0 = date_shift_month(test_date_start, lag)
date_t1 = date_shift_month(test_date_end, lag)
loans_yoy_change_interval_test = get_interval_data(comm_indus_loans_df, 'loans_yoy_change', date_t0, date_t1)

#PPI
lag = -1
date_t0 = date_shift_month(test_date_start, lag)
date_t1 = date_shift_month(test_date_end, lag)
ppi_yoy_change_interval_test = get_interval_data(ppi_df, 'ppi_yoy_change', date_t0, date_t1)

#uemp
lag = -1
date_t0 = date_shift_month(test_date_start, lag)
date_t1 = date_shift_month(test_date_end, lag)
unemp_interval_test = get_interval_data(unemp_df, 'Unemployment Rate', date_t0, date_t1)
unemp_monthly_change_interval_test = get_interval_data(unemp_df, 'unemp_monthly_change', date_t0, date_t1)

#dgo
lag = -2
date_t0 = date_shift_month(test_date_start, lag)
date_t1 = date_shift_month(test_date_end, lag)
dgo_mom_change_interval_test = get_interval_data(dgo_df, 'dgo_mom_change', date_t0, date_t1)


#PI
lag = -2
date_t0 = date_shift_month(test_date_start, lag)
date_t1 = date_shift_month(test_date_end, lag)
pi_yoy_change_interval_test = get_interval_data(pi_df, 'pi_yoy_change', date_t0, date_t1)

#PCE
lag = -2
date_t0 = date_shift_month(test_date_start, lag)
date_t1 = date_shift_month(test_date_end, lag)
pcepi_yoy_change_interval_test = get_interval_data(pcepi_df, 'pcepi_yoy_change', date_t0, date_t1)



X_test = [loans_yoy_change_interval_test, ppi_yoy_change_interval_test, 
       unemp_interval_test, unemp_monthly_change_interval_test, dgo_mom_change_interval_test,
      pi_yoy_change_interval_test, pcepi_yoy_change_interval_test]

X_test = np.array(X_test)
X_test = X_test.transpose()

X_poly_test = poly.fit_transform(X_test)
X_poly_test = sm.add_constant(X_poly_test)

y_esti_test = results.predict(X_poly_test)

figure2 = plt.figure()
time = range(0,len(y_esti_test))
plt.plot(time,bank_mom_change_interval_test,c="r",label='real stock returns of test set')
plt.plot(time,y_esti_test,c="b",label='estimated stock returns')
plt.xlabel("month index",fontsize=12)
plt.ylabel('stock returns', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=12)
# plt.show()

mse_train = mean_squared_error(y_real=bank_mom_change_interval, y_esti=y_esti_train)
mse_test = mean_squared_error(bank_mom_change_interval_test, y_esti_test)

print(mse_train, mse_test)


## step 4 : ridge regression
alpha_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 10]
mse_test = []
mse_train = []
y_train = bank_mom_change_interval
y_test = bank_mom_change_interval_test

for i in range(0, len(alpha_list)):
    ridge_reg_result = Ridge(alpha_list[i],fit_intercept=False).fit(X_poly_train, y_train)
    y_estimate_train_ridge = ridge_reg_result.predict(X_poly_train)
    y_estimate_test_ridge = ridge_reg_result.predict(X_poly_test)
    mse_train.append(mean_squared_error(y_train, y_estimate_train_ridge))
    mse_test.append(mean_squared_error(y_test, y_estimate_test_ridge))


print(np.around(np.array(mse_train),4))
print(np.around(np.array(mse_test),4))

figure3 = plt.figure()
x = range(0, len(alpha_list))
plt.plot(x, mse_test, label='test data')
plt.plot(x,mse_train,label='training data')
plt.xticks(x, alpha_list)
plt.xlabel("Value of alpha")
plt.ylabel("Mean squared error")
plt.legend()



ridge_reg_result = Ridge(0.5,fit_intercept=False).fit(X_poly_train, y_train)
y_estimate_train_ridge = ridge_reg_result.predict(X_poly_train)
y_estimate_test_ridge = ridge_reg_result.predict(X_poly_test)


figure4 = plt.figure()
time = range(0,len(y_train))
plt.plot(time,bank_mom_change_interval,c="r",label='real stock returns')
plt.plot(time,y_estimate_train_ridge,c="b",label='estimated stock returns')
plt.xlabel("month index")
plt.ylabel('stock returns')
plt.legend()

figure5 = plt.figure()
time = range(0,len(y_test))
plt.plot(time,bank_mom_change_interval_test,c="r",label='real stock returns')
plt.plot(time,y_estimate_test_ridge,c="b",label='estimated stock returns')
plt.xlabel("month index")
plt.ylabel('stock returns')
plt.legend()
plt.show()