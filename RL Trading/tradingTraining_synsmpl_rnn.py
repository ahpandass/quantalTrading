import pandas as pd
import numpy as np
import talib,json,requests



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

#s = kline(symbol,startTime,endTime)



import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
#import matplotlib.finance as mpf
from matplotlib.pylab import date2num
#from DataUtils import *
import talib
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
%tensorflow_version 1.x
from tqdm import tqdm
import seaborn as sns
import os

symbol = 'BNBUSDT'
startTime = '2021-10-01'
endTime = '2022-01-03'
interval ='30m'
train_round = 10000
model_path = '.'
freq_cnt = 1440
window = 20


lmap=lambda func,it: map(lambda x:func(x),it)
lfilter=lambda func,it: filter(lambda x:func(x),it)
z_score=lambda x:(x-np.mean(x,axis=0))/(np.std(x,axis=0)+1e-5)

class DRL_Crypto_portfolio(object):
    def __init__(self, feature_number, action_size=1,c=1e-5, hidden_units_number=[128, 64], learning_rate=0.001):
        tf.reset_default_graph()
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, feature_number], name='s')
        self.d = tf.placeholder(dtype=tf.float32, shape=[None,action_size-1], name='d')
        self.s_buffer=[]
        self.d_buffer=[]
        self.c=c
        self.action_size=action_size
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('rnn_encoder', initializer=tf.keras.initializers.glorot_normal, regularizer=tf.keras.regularizers.L2(0.01)):
#             cell=self._add_GRU(units_number=128,keep_prob=self.dropout_keep_prob)
            cells=self._add_GRUs(units_number=[128,action_size],activation=[tf.nn.relu,tf.nn.relu])
            self.rnn_input=tf.expand_dims(self.s,axis=0)
            self.rnn_output,_=tf.nn.dynamic_rnn(inputs=self.rnn_input,cell=cells,dtype=tf.float32)
#             self.rnn_output=tf.contrib.layers.layer_norm(self.rnn_output)
            self.a_prob=tf.unstack(self.rnn_output,axis=0)[0]
            
#         with tf.variable_scope('supervised',initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
#             self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
#             self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=[feature_number], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
#             self.state_loss=tf.losses.mean_squared_error(self.state_predict,self.s_next)
            
        with tf.variable_scope('direct_RL',initializer=tf.keras.initializers.glorot_normal, regularizer=tf.keras.regularizers.L2(0.01)):
#             self.rnn_output=tf.stop_gradient(self.rnn_output)
#             self.a_prob = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number+[action_size], drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
#             self.a_prob = self._add_dense_layer(inputs=self.a_prob, output_shape=, drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.a_out = tf.nn.softmax(self.a_prob,axis=-1)
            self.a_out = tf.concat((tf.zeros(dtype=tf.float32,shape=[1,self.action_size]), self.a_out), axis=0)
            self.reward = tf.reduce_sum(self.d * self.a_out[:-1,:-1] - self.c * tf.abs(self.a_out[1:,:-1] - self.a_out[:-1,:-1]),axis=1)
            self.total_reward = tf.reduce_sum(self.reward)
            self.mean_reward = tf.reduce_mean(self.reward)
            
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(-self.mean_reward)
        self.init_op = tf.global_variables_initializer()
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        
    def init_model(self):
        self.session.run(self.init_op)
    
    def _add_dense_layer(self, inputs, output_shape, drop_keep_prob, act=tf.nn.relu, use_bias=True):
        output = inputs
        for n in output_shape:
            output = tf.layers.dense(output, n, activation=act, use_bias=use_bias)
            output = tf.nn.dropout(output, drop_keep_prob)
        return output
    
    def _add_GRU(self,units_number,activation=tf.nn.relu,keep_prob=1.0):
        cell = tf.nn.rnn_cell.LSTMCell(units_number,activation=activation)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
        return cell
    
    def _add_GRUs(self,units_number,activation,keep_prob=1.0):
        cells=tf.nn.rnn_cell.MultiRNNCell(cells=[ self._add_GRU(units_number=n,activation=a) for n,a in zip(units_number,activation)])
        return cells
    
    def _add_gru_cell(self, units_number, activation=tf.nn.relu):
        return tf.contrib.rnn.GRUCell(num_units=units_number, activation=activation)
    
    def train(self, drop=0.85):
#         np.random.shuffle(random_index)
        feed = {
            self.s: np.array(self.s_buffer),
            self.d: np.array(self.d_buffer),
            self.dropout_keep_prob: drop
        }
        self.session.run([self.train_op], feed_dict=feed)
    
    def restore_buffer(self):
        self.s_buffer = []
        self.d_buffer = []
    def save_current_state(self,s,d):
        self.s_buffer.append(s)
        self.d_buffer.append(d)
    
    def trade(self,train=False, drop=1.0, prob=False):
        feed = {
            self.s: np.array(self.s_buffer),
            self.dropout_keep_prob: drop
        }
        a_prob = self.session.run([self.a_out], feed_dict=feed)
        a_prob = a_prob[-1][-1].flatten()
        return a_prob
    def load_model(self, model_path='./DRLModel'):
        self.saver.restore(self.session, model_path + '/model')

    def save_model(self, model_path='./DRLModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)

def get_rand(normalize_length, train_length, assets_length):
  return np.random.randint(low=normalize_length, high= assets_length - train_length)

assets=[symbol]
asset_data=lfilter(lambda x:x[1] is not None,lmap(lambda x:(x,kline(x,startTime,endTime,interval=interval,count=500)),assets))
asset_data=lmap(lambda x:(x[0],generate_tech_data(x[1],close_name='close',high_name='high',low_name='low',open_name='open')),asset_data)
asset_data=dict(asset_data)
#asset_data=pd.Panel(asset_data)
asset_data_t = []
for key in asset_data.keys():
  asset_data_t.append(np.array(asset_data[key]))

asset_data = np.array(asset_data_t)
model=DRL_Crypto_portfolio(action_size=asset_data.shape[0]+1,feature_number=asset_data.shape[2]*asset_data.shape[0],learning_rate=1e-3)
#model.load_model()
model.init_model()
model.restore_buffer()
normalize_length_base=20
train_length_base=1500
batch_size=64
c=1e-3
epoch=300
train_r=[]
train_mean_r=[]
test_r=[]
test_mean_r=[]
for e in range(epoch):
    train_reward=[]
    train_actions=[]
    test_reward=[]
    test_actions=[]
    normalize_length = get_rand(normalize_length_base, train_length_base, asset_data.shape[1])
    train_length = normalize_length + train_length_base
    previous_action=np.zeros(asset_data.shape[0]+1)
    for t in range(normalize_length,train_length):
        state=asset_data[:,t-normalize_length:t,:]
        #diff=state[:,:,2][-1]
        #diff = state[:,:,2][0][-1]
        diff=np.expand_dims(state[:,:,2][0][-1],axis=-1)
        #print(state[:,:,2])
        #state=state.values
        state_pre = state
        state=state.reshape((state.shape[1],state.shape[0]*state.shape[2]))
        state=z_score(state)[None,-1]
        model.save_current_state(s=state[0],d=diff)
        action=model.trade(state)
        r=np.sum(asset_data[:,:,2][-1][t]*action[:-1]-c*np.sum(np.abs(previous_action-action)))
        previous_action=action
        train_reward.append(r)
        train_actions.append(action)
        if t % batch_size == 0:
            model.train(drop=0.8)
            model.restore_buffer()
    print(e,'train_reward',np.sum(train_reward),np.mean(train_reward))
    train_r.append(np.sum(train_reward))
    train_mean_r.append(np.mean(train_reward))
    previous_action=np.zeros(asset_data.shape[0]+1)

    normalize_length = get_rand(normalize_length_base, train_length, asset_data.shape[1])
    train_length = normalize_length + train_length_base

    for t in range(normalize_length,train_length):
        state=asset_data[:,t-normalize_length:t,:]
        #diff = state[:,:,2][0][-1]
        diff=np.expand_dims(state[:,:,2][0][-1],axis=-1)
        #state=state.values
        state=state.reshape((state.shape[1],state.shape[0]*state.shape[2]))
        state=z_score(state)[None,-1]
        model.save_current_state(s=state[0],d=diff)
        action=model.trade(state)
        r=np.sum(asset_data[:,:,2][-1][t]*action[:-1]-c*np.sum(np.abs(previous_action-action)))
        test_reward.append(r)
        test_actions.append(action)
        previous_action=action
        if t % batch_size==0:
            model.train(drop=0.8)
            model.restore_buffer()
    print(e,'test_reward',np.sum(test_reward),np.mean(test_reward))
    model.save_model()
    test_r.append(np.sum(test_reward))
    test_mean_r.append(np.mean(test_reward))
    model.restore_buffer()
    if np.sum(np.sum(test_reward))>0.6: break
model.restore_buffer()