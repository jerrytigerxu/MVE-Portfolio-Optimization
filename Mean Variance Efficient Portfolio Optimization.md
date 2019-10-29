
# Finding the Mean Variance Efficient Portfolio

In order to find the portfolio that gives us the best returns and ultimately the most money, we need to create a framework that allows us to minimize volatility and/or maximize the Sharpe Ratio.

## Data

To easily download stock price data, we will use the DataReader library in Pandas. 


```
import pandas_datareader as web
import pandas as pd
import numpy as np
```

Let's quickly test to see that the library is working properly by downloading some Tesla stock data.


```
tslaQuote = web.DataReader('TSLA', data_source='yahoo', start='2010-01-01', end='2018-12-31')
```


```
print(tslaQuote)

# Looks like it works!
```

                      High         Low  ...    Volume   Adj Close
    Date                                ...                      
    2010-06-29   25.000000   17.540001  ...  18766300   23.889999
    2010-06-30   30.420000   23.299999  ...  17187100   23.830000
    2010-07-01   25.920000   20.270000  ...   8218800   21.959999
    2010-07-02   23.100000   18.709999  ...   5139800   19.200001
    2010-07-06   20.000000   15.830000  ...   6866900   16.110001
    2010-07-07   16.629999   14.980000  ...   6921700   15.800000
    2010-07-08   17.520000   15.570000  ...   7711400   17.459999
    2010-07-09   17.900000   16.549999  ...   4050600   17.400000
    2010-07-12   18.070000   17.000000  ...   2202500   17.049999
    2010-07-13   18.639999   16.900000  ...   2680100   18.139999
    2010-07-14   20.150000   17.760000  ...   4195200   19.840000
    2010-07-15   21.500000   19.000000  ...   3739800   19.889999
    2010-07-16   21.299999   20.049999  ...   2621300   20.639999
    2010-07-19   22.250000   20.920000  ...   2486500   21.910000
    2010-07-20   21.850000   20.049999  ...   1825300   20.299999
    2010-07-21   20.900000   19.500000  ...   1252500   20.219999
    2010-07-22   21.250000   20.370001  ...    957800   21.000000
    2010-07-23   21.559999   21.059999  ...    653600   21.290001
    2010-07-26   21.500000   20.299999  ...    922200   20.950001
    2010-07-27   21.180000   20.260000  ...    619700   20.549999
    2010-07-28   20.900000   20.510000  ...    467200   20.719999
    2010-07-29   20.879999   20.000000  ...    616000   20.350000
    2010-07-30   20.440001   19.549999  ...    426900   19.940001
    2010-08-02   20.969999   20.330000  ...    718100   20.920000
    2010-08-03   21.950001   20.820000  ...   1230500   21.950001
    2010-08-04   22.180000   20.850000  ...    913000   21.260000
    2010-08-05   21.549999   20.049999  ...    796200   20.450001
    2010-08-06   20.160000   19.520000  ...    741900   19.590000
    2010-08-09   19.980000   19.450001  ...    812700   19.600000
    2010-08-10   19.650000   18.820000  ...   1281300   19.030001
    ...                ...         ...  ...       ...         ...
    2018-11-15  348.579987  339.040009  ...   4625700  348.440002
    2018-11-16  355.700012  345.119995  ...   7206200  354.309998
    2018-11-19  366.750000  352.880005  ...   9708900  353.470001
    2018-11-20  349.799988  333.549988  ...   8004700  347.489990
    2018-11-21  353.100006  337.399994  ...   4686800  338.190002
    2018-11-23  337.500000  325.549988  ...   4202600  325.829987
    2018-11-26  346.220001  325.000000  ...   7992100  346.000000
    2018-11-27  346.959991  335.500000  ...   6358300  343.920013
    2018-11-28  348.279999  342.209991  ...   4127600  347.869995
    2018-11-29  347.500000  339.549988  ...   3080700  341.170013
    2018-11-30  351.600006  338.260010  ...   5629100  350.480011
    2018-12-03  366.000000  352.000000  ...   8306500  358.489990
    2018-12-04  368.679993  352.000000  ...   8461900  359.700012
    2018-12-06  367.380005  350.760010  ...   7842500  363.059998
    2018-12-07  379.489990  357.649994  ...  11511200  357.970001
    2018-12-10  365.980011  353.119995  ...   6613500  365.149994
    2018-12-11  372.170013  360.230011  ...   6308800  366.760010
    2018-12-12  371.910004  365.160004  ...   5027000  366.600006
    2018-12-13  377.440002  366.750000  ...   7365900  376.790009
    2018-12-14  377.869995  364.329987  ...   6337600  365.709991
    2018-12-17  365.700012  343.880005  ...   7674000  348.420013
    2018-12-18  351.549988  333.690002  ...   7100000  337.029999
    2018-12-19  347.010010  329.739990  ...   8274200  332.970001
    2018-12-20  330.290009  311.869995  ...   9071900  315.380005
    2018-12-21  323.470001  312.440002  ...   8016800  319.769989
    2018-12-24  314.500000  295.200012  ...   5559900  295.390015
    2018-12-26  326.970001  294.089996  ...   8163100  326.089996
    2018-12-27  322.170013  301.500000  ...   8575100  316.130005
    2018-12-28  336.239990  318.410004  ...   9939000  333.869995
    2018-12-31  339.209991  325.260010  ...   6302300  332.799988
    
    [2142 rows x 6 columns]


Let's quickly test out calculating portfolio mean returns and variances. We'll make a momentous jump from one stock to four and calculate some stuff.


```
stocks = ['GOOGL', 'TM', 'KO', 'PEP']
numAssets = len(stocks)
source = 'yahoo'
start = '2010-01-01'
end = ' 2019-5-31'


```


```
import pandas as pd
import numpy as np
data = pd.DataFrame(columns=stocks)
for symbol in stocks:
  data[symbol] = web.DataReader(symbol, data_source=source, start=start, end=end)['Adj Close']
 
```


```
list(data)
data['GOOGL']

# Here is the value given the key of 'GOOGL'
```




    Date
    2010-01-04     313.688690
    2010-01-05     312.307312
    2010-01-06     304.434448
    2010-01-07     297.347351
    2010-01-08     301.311310
    2010-01-11     300.855865
    2010-01-12     295.535522
    2010-01-13     293.838837
    2010-01-14     295.220215
    2010-01-15     290.290283
    2010-01-19     294.104095
    2010-01-20     290.495483
    2010-01-21     291.781769
    2010-01-22     275.280273
    2010-01-25     270.270264
    2010-01-26     271.481476
    2010-01-27     271.321320
    2010-01-28     267.412415
    2010-01-29     265.235229
    2010-02-01     266.776764
    2010-02-02     265.825836
    2010-02-03     270.680695
    2010-02-04     263.653656
    2010-02-05     265.910919
    2010-02-08     267.002014
    2010-02-09     268.488495
    2010-02-10     267.492493
    2010-02-11     268.468475
    2010-02-12     266.826813
    2010-02-16     270.920929
                     ...     
    2019-04-18    1241.469971
    2019-04-22    1253.760010
    2019-04-23    1270.589966
    2019-04-24    1260.050049
    2019-04-25    1267.339966
    2019-04-26    1277.420044
    2019-04-29    1296.199951
    2019-04-30    1198.959961
    2019-05-01    1173.319946
    2019-05-02    1166.510010
    2019-05-03    1189.550049
    2019-05-06    1193.459961
    2019-05-07    1178.859985
    2019-05-08    1170.780029
    2019-05-09    1167.969971
    2019-05-10    1167.640015
    2019-05-13    1136.589966
    2019-05-14    1124.859985
    2019-05-15    1170.800049
    2019-05-16    1184.500000
    2019-05-17    1168.780029
    2019-05-20    1144.660034
    2019-05-21    1154.439941
    2019-05-22    1155.849976
    2019-05-23    1145.339966
    2019-05-24    1138.609985
    2019-05-28    1139.560059
    2019-05-29    1119.939941
    2019-05-30    1121.410034
    2019-05-31    1106.500000
    Name: GOOGL, Length: 2368, dtype: float64



## Calculating Returns and Volatility Based on Historical Data
For returns, we'll calculate the logarithmic returns.


```
# Calculting log returns
percent_change = data.pct_change()
returns = np.log(1+percent_change)
```


```
returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOOGL</th>
      <th>TM</th>
      <th>KO</th>
      <th>PEP</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-04</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>-0.004413</td>
      <td>-0.015517</td>
      <td>-0.012170</td>
      <td>0.012011</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>-0.025532</td>
      <td>0.012692</td>
      <td>-0.000355</td>
      <td>-0.010054</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>-0.023555</td>
      <td>-0.012453</td>
      <td>-0.002489</td>
      <td>-0.006376</td>
    </tr>
    <tr>
      <th>2010-01-08</th>
      <td>0.013243</td>
      <td>0.023239</td>
      <td>-0.018682</td>
      <td>-0.003286</td>
    </tr>
  </tbody>
</table>
</div>



Now let's get the mean daily returns and the covariance matrix.


```
meanDailyReturns = returns.mean()
covMatrix = returns.cov()
```


```
meanDailyReturns

# Returns you are getting on average per day from these companies
```




    GOOGL    0.000533
    TM       0.000244
    KO       0.000382
    PEP      0.000430
    dtype: float64




```
meanReturns = meanDailyReturns * 365
```


```
meanReturns
```




    GOOGL    0.194382
    TM       0.089113
    KO       0.139328
    PEP      0.157087
    dtype: float64




```
covMatrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOOGL</th>
      <th>TM</th>
      <th>KO</th>
      <th>PEP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GOOGL</th>
      <td>0.000234</td>
      <td>0.000072</td>
      <td>0.000045</td>
      <td>0.000042</td>
    </tr>
    <tr>
      <th>TM</th>
      <td>0.000072</td>
      <td>0.000174</td>
      <td>0.000043</td>
      <td>0.000040</td>
    </tr>
    <tr>
      <th>KO</th>
      <td>0.000045</td>
      <td>0.000043</td>
      <td>0.000087</td>
      <td>0.000055</td>
    </tr>
    <tr>
      <th>PEP</th>
      <td>0.000042</td>
      <td>0.000040</td>
      <td>0.000055</td>
      <td>0.000080</td>
    </tr>
  </tbody>
</table>
</div>



Let's create a hypothetical portfolio and see what returns and standard deviation we get.

For allocation, we'll do: Google - 50%, Toyota - 20%, Coca Cola - 20%, Pepsi - 10%


```
# Calculate expected portfolio performance

weights= np.array([0.5, 0.2, 0.2, 0.1])
portReturn = np.sum(meanDailyReturns*weights)
portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
```

### IMPORTANT: Keep in mind that what we are getting is DAILY numbers


```
portReturn

```




    0.00043448745388989876




```
portReturn*365

# Our yearly returns
```




    0.15858792066981306




```
portStdDev
```




    0.010236332166865772




```
portStdDev*portStdDev*365

# Our yearly standard deviation
```




    0.038245611124099986



Based on these numbers, and given the data from between 2010 and 2019, we're most likely to get a yearly return of between 12% and 20%. Not bad, huh?

### Visualizing the Data

Let's see how all of the company stock data compares to each other


```
import matplotlib.pyplot as plt
```


```
plt.figure(figsize=(14, 7))
for c in returns.columns.values:
  plt.plot(returns.index, returns[c], lw=3, alpha=0.8, label=c)

plt.legend(loc='upper left', fontsize=12)
plt.ylabel('returns')
```

    /usr/local/lib/python3.6/dist-packages/pandas/plotting/_converter.py:129: FutureWarning: Using an implicitly registered datetime converter for a matplotlib plotting method. The converter was registered by pandas on import. Future versions of pandas will require you to explicitly register matplotlib converters.
    
    To register the converters:
    	>>> from pandas.plotting import register_matplotlib_converters
    	>>> register_matplotlib_converters()
      warnings.warn(msg, FutureWarning)





    Text(0, 0.5, 'returns')




![png](Mean%20Variance%20Efficient%20Portfolio%20Optimization_files/Mean%20Variance%20Efficient%20Portfolio%20Optimization_28_2.png)


## Defining Functions

Now that we've established the basics of portfolio optimization, we will now use some advanced methods to calculate the optimal portfolio allocations based on our returns and volatility information.



```
import scipy.optimize as sco
```

Scipy has an optimize function that can perform very complex calculations that would take us forever to do by hand


```
def calcPortfolioPerf(weights, meanReturns, covMatrix):
  
  portReturn = np.sum(meanReturns*weights)
  portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
  return portReturn, portStdDev
```

Some as what did before, just encapsulated into a function. It outputs two parameters, the portfolio return and the portfolio standard deviation given three arguments: a list of weights, a list of mean returns, and the covariance matrix


```
def negSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate):
  
  p_ret, p_var = calcPortfolioPerf(weights, meanReturns, covMatrix)
  
  return -(p_ret - riskFreeRate) / p_var
    
    
```

The reason why we are calculating the negative Sharpe Ratio is because scipy can only minimize with its optimization function, and we want to maximize the Sharpe Ratio, so we'll make it negative.

If you don't already know, the Sharpe Ratio is simply the risk premium (which is the amount of return you get for bearing risk), divided by the standard deviation of the portfolio.

To put it succintly, the Sharpe Ratio is how much returns you get relative to how much risk you put in. 


```
def getPortfolioVol(weights, meanReturns, covMatrix):
  return calcPortfolioPerf(weights, meanReturns, covMatrix)[1]
```

This is the same thing as the CalcPortfolioPerf() function, just taking the volatility parameter.


```
# Finding the portfolio with the maximum Sharpe Ratio using scipy instead of monte carlo method

def findMaxSharpeRatioPortfolio(meanReturns, covMatrix, riskFreeRate):
  
  numAssets = len(meanReturns)
  args = (meanReturns, covMatrix, riskFreeRate)
  constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
  bounds = tuple((0,1) for asset in range(numAssets))
  
  opts = sco.minimize(negSharpeRatio, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
  
  return opts
```


```
findMaxSharpeRatioPortfolio(meanReturns, covMatrix, 0.03)
```




         fun: -15.8453589769726
         jac: array([-3.58823228,  0.1951772 , -3.57979631, -3.57948232])
     message: 'Optimization terminated successfully.'
        nfev: 45
         nit: 7
        njev: 7
      status: 0
     success: True
           x: array([2.31074402e-01, 1.03115112e-15, 1.67265407e-01, 6.01660191e-01])



This function is where we actually use the scipy minimize function. It's very similar to the solver function in Excel. Given the constraints, bounds, and arguments, the function will spit out the optimal allocation that will minimize the negative Sharpe Ratio, which is the same thing as maximizing the Sharpe Ratio.


```
def findEfficientReturn(meanReturns, covMatrix, targetReturn):
  numAssets = len(meanReturns)
  args = (meanReturns, covMatrix)
  
  def getPortfolioReturn(weights):
    return calcPortfolioPerf(weights, meanReturns, covMatrix)[0]
  
  constraints = ({'type': 'eq', 'fun': lambda x: getPortfolioReturn(x) - targetReturn},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
  bounds = tuple((0,1) for asset in range(numAssets))
  
  return sco.minimize(getPortfolioVol, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
```

This is very similar to the previous function, except the goal is not to maximize the Sharpe Ratio, rather the goal is to minimize volatility.


```
def findEfficientFrontier(meanReturns, covMatrix, rangeOfReturns):
  efficientPortfolios = []
  for ret in rangeOfReturns:
    efficientPortfolios.append(findEfficientReturn(meanReturns, covMatrix, ret))
    
  return efficientPortfolios
```


```
findEfficientReturn(meanReturns, covMatrix, 0.1)
```




         fun: 0.011201820104480014
         jac: array([0.00594181, 0.01300592, 0.00468494, 0.00385015])
     message: 'Optimization terminated successfully.'
        nfev: 30
         nit: 5
        njev: 5
      status: 0
     success: True
           x: array([0.        , 0.78318684, 0.21681316, 0.        ])



## Monte Carlo Simulation


```
plt.figure(figsize=(8,6))
```




    <Figure size 576x432 with 0 Axes>




    <Figure size 576x432 with 0 Axes>



```
numPortfolios = 100000
results = np.zeros((3, numPortfolios))


```


```
stocks = ['GOOGL', 'TM', 'KO', 'PEP']
numAssets = len(stocks)
source = 'yahoo'
start = '2010-01-01'
end = '2019-05-31'

data = pd.DataFrame(columns=stocks)
for symbol in stocks:
  data[symbol] = web.DataReader(symbol, data_source=source, start=start, end=end)['Adj Close']

riskFreeRate = 0.0021
dur = 20
numPeriodsAnnually = 252.0/dur
windowedData = data[::dur]
rets = np.log(windowedData/windowedData.shift(1))
```


```
meanDailyReturn = rets.mean()
covariance = rets.cov()
```


```
for i in range(numPortfolios):
  weights = np.random.random(numAssets)
  weights /= np.sum(weights)
  
  pret, pvar = calcPortfolioPerf(weights, meanDailyReturn, covariance)
  
  results[0,i] = pret*numPeriodsAnnually
  results[1,i] = pvar*np.sqrt(numPeriodsAnnually)
  results[2,i] = (results[0,i] - riskFreeRate)/results[1,i]
```


```
plt.scatter(results[1,:], results[0,:], c=results[2,:], marker='o')
```




    <matplotlib.collections.PathCollection at 0x7fe194ca0358>




![png](Mean%20Variance%20Efficient%20Portfolio%20Optimization_files/Mean%20Variance%20Efficient%20Portfolio%20Optimization_51_1.png)



```
targetReturns = np.linspace(0.09, 0.26, 50)/(252./dur)
efficientPortfolios = findEfficientFrontier(meanDailyReturn, covariance, targetReturns)
plt.plot([p['fun']*np.sqrt(numPeriodsAnnually) for p in efficientPortfolios], targetReturns*numPeriodsAnnually, marker='x')
```




    [<matplotlib.lines.Line2D at 0x7fe194d5e588>]




![png](Mean%20Variance%20Efficient%20Portfolio%20Optimization_files/Mean%20Variance%20Efficient%20Portfolio%20Optimization_52_1.png)



```

```
