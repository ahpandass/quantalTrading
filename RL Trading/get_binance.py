import pandas as pd
import numpy as np
import talib,json,requests

symbol = 'BNBUSDT'
startTime = '2022-01-01'
endTime = '2022-01-03'
interval ='1m'
train_round = 10000
model_path = '.'
freq_cnt = 1440
window = 20

import datetime
def get_binace(symbol, interval, startTime, endTime):
    df_list = []
    symbol = symbol.replace('/','')
    since_datetime = datetime.datetime(int(startTime[0:4]),int(startTime[5:7]),int(startTime[8:10]))
    end_datetime = datetime.datetime(int(endTime[0:4]),int(endTime[5:7]),int(endTime[8:10]))
    while True:
        new_df = get_binance_bars(symbol,interval, since_datetime, end_datetime)
        if new_df is None:
            break
        df_list.append(new_df)
        since_datetime= max(new_df.index) + datetime.timedelta(0,1)
        
    df = pd.concat(df_list)
    return df

def get_binance_bars(symbol, interval, startTime, endTime):
    url = 'https://api.binance.com/api/v3/klines'
    startTime = str(int(startTime.timestamp()*1000))
    endTime = str(int(endTime.timestamp()*1000))
    limit = '1000'
    req_params = {'symbol':symbol, 'interval':interval, 'startTime':startTime, 'endTime':endTime, 'limit':limit}
    df = pd.DataFrame(json.loads(requests.get(url, params=req_params).text))
    print('finish 1 round, get {} records'.format(len(df)))
    if (len(df.index)==0):
        return None
    
    df = df.iloc[:,0:5]
    df.columns=['datetime','open','high','low','close']
    
    df.open = df.open.astype('float')
    df.high = df.high.astype('float')
    df.low = df.low.astype('float')
    df.close = df.close.astype('float')
    #df.volume = df.volume.astype('float')
    
    df.index = [datetime.datetime.fromtimestamp(x/1000.0) for x in df.datetime]
    return df

def generate_tech_data(stock, open_name, close_name, high_name, low_name, max_time_window=10):
    open_price = stock[open_name].values
    close_price = stock[close_name].values
    low_price = stock[low_name].values
    high_price = stock[high_name].values
    data = stock.copy()
    data['MOM'] = talib.MOM(close_price, timeperiod=max_time_window)
    data['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)
    data['HT_DCPHASE'] = talib.HT_DCPHASE(close_price)
    data['sine'], data['leadsine'] = talib.HT_SINE(close_price)
    data['inphase'], data['quadrature'] = talib.HT_PHASOR(close_price)
    data['ADXR'] = talib.ADXR(high_price, low_price, close_price, timeperiod=max_time_window)
    data['APO'] = talib.APO(close_price, fastperiod=max_time_window // 2, slowperiod=max_time_window)
    data['AROON_UP'], _ = talib.AROON(high_price, low_price, timeperiod=max_time_window)
    data['CCI'] = talib.CCI(high_price, low_price, close_price, timeperiod=max_time_window)
    data['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=max_time_window)
    data['PPO'] = talib.PPO(close_price, fastperiod=max_time_window // 2, slowperiod=max_time_window)
    data['macd'], data['macd_sig'], data['macd_hist'] = talib.MACD(close_price, fastperiod=max_time_window // 2, slowperiod=max_time_window, signalperiod=max_time_window // 2)
    data['CMO'] = talib.CMO(close_price, timeperiod=max_time_window)
    data['ROCP'] = talib.ROCP(close_price, timeperiod=max_time_window)
    data['fastk'], data['fastd'] = talib.STOCHF(high_price, low_price, close_price)
    data['TRIX'] = talib.TRIX(close_price, timeperiod=max_time_window)
    data['ULTOSC'] = talib.ULTOSC(high_price, low_price, close_price, timeperiod1=max_time_window // 2, timeperiod2=max_time_window, timeperiod3=max_time_window * 2)
    data['WILLR'] = talib.WILLR(high_price, low_price, close_price, timeperiod=max_time_window)
    data['NATR'] = talib.NATR(high_price, low_price, close_price, timeperiod=max_time_window)
    data = data.drop([open_name, close_name, high_name, low_name], axis=1)
    data = data.dropna().astype(np.float32)
    return data


def kline(symbol,startTime, endTime, interval='1m', count=500):
    s = get_binace(symbol,interval,startTime,endTime)
    if s is None: return None
    s = pd.DataFrame(s)[::-1]
    if s.shape[0] < count:
        return None
    s['avg'] = (np.mean(s[['open', 'high', 'low', 'close']], axis=1))
    s['diff'] = np.log(s['avg'] / s['avg'].shift(1)).fillna(0)
    return s
