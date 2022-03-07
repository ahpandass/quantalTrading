
# -*- coding: utf-8 -*-

import logging,requests,datetime, json
import tempfile

import numpy as np
import pandas as pd
from talib.abstract import *

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

ret = lambda x, y: np.log(y / x)  # Log return
zscore = lambda x: (x - np.nanmean(x)) / np.nanstd(x)  # zscore

class ZiplineEnvSrc(object):
    # Quandl-based implementation of a TradingEnv's data source.
    # Pulls data from Quandl, preps for use by TradingEnv and then
    # acts as data provider for each new episode.
    def __init__(self, symbol, interval, start, end, freq_cnt=1440, scale=True):
        self.symbol = symbol
        self.freq_cnt = freq_cnt + 1
        self.start = start
        self.end = end

        log.info('getting data for %s from binance marking...', symbol)
        
        _df = self.get_binace(symbol, interval, start, end) #need code, only need [datetime, close], but adding volumn in next step

        df = pd.DataFrame()
        ##########################
        close = _df['close'].values
        sma5 = SMA(_df['close'], timeperiod=5)
        sma15 = SMA(_df['close'], timeperiod=15)
        rsi = RSI(_df['close'], timeperiod=5)
        atr = ATR(_df['high'],_df['low'],_df['close'], timeperiod=5)

        df['Return'] = (_df.close - _df.close.shift()) / _df.close.shift()  # today return
        df['SMA5'] = zscore(sma5)
        df['SMA15'] = zscore(sma15)
        df['C-SMA5'] = zscore(close - sma5)
        df['SMA5-SMA15'] = zscore(sma5 - sma15)
        df['RSI'] = zscore(rsi)
        df['ATR'] = zscore(atr)
        df['VOL'] = zscore(_df.volume)
        df['CLOSE'] = zscore(_df.close)

        df.dropna(axis=0, inplace=True)

        self.data = df
        self.step = 0
        self.orgin_idx = 0
    
    def get_binace(self, symbol, interval, startTime, endTime):
        df_list = []
        symbol = symbol.replace('/','')
        since_datetime = datetime.datetime(int(startTime[0:4]),int(startTime[5:7]),int(startTime[8:10]))
        end_datetime = datetime.datetime(int(endTime[0:4]),int(endTime[5:7]),int(endTime[8:10]))
        while True:
            new_df = self.get_binance_bars(symbol,interval, since_datetime, end_datetime)
            if new_df is None:
                break
            df_list.append(new_df)
            since_datetime= max(new_df.index) + datetime.timedelta(0,1)
            
        df = pd.concat(df_list)
        return df
    
    def get_binance_bars(self, symbol, interval, startTime, endTime):
        url = 'https://api.binance.com/api/v3/klines'
        startTime = str(int(startTime.timestamp()*1000))
        endTime = str(int(endTime.timestamp()*1000))
        limit = '1000'
        req_params = {'symbol':symbol, 'interval':interval, 'startTime':startTime, 'endTime':endTime, 'limit':limit}
        print(req_params)
        df = pd.DataFrame(json.loads(requests.get(url, params=req_params).text))
        print('finish 1 round, get {} records'.format(len(df)))
        if (len(df.index)==0):
            return None
        
        df = df.iloc[:,0:6]
        df.columns=['datetime','open','high','low','close','volume']
        
        df.open = df.open.astype('float')
        df.high = df.high.astype('float')
        df.low = df.low.astype('float')
        df.close = df.close.astype('float')
        df.volume = df.volume.astype('float')
        
        df.index = [datetime.datetime.fromtimestamp(x/1000.0) for x in df.datetime]
        return df
    
    def reset(self,random):
        if random == True:
            self.idx = np.random.randint(low=0, high=len(self.data.index) - self.freq_cnt)
        else:
            self.idx = len(self.data.index) - self.freq_cnt
        self.step = 0
        self.orgin_idx = self.idx  # for render , so record it
        self.reset_start_day = str(self.data.index[self.orgin_idx -1 ])[:10]
        self.reset_end_day = str(self.data.index[self.orgin_idx + self.freq_cnt -1 ])[:10]
        #print(self.reset_start_day,self.reset_end_day)


    def _step(self):
        obs = list(self.data.iloc[self.idx])
        self.idx += 1
        self.step += 1
        done = self.step > self.freq_cnt
        return obs, done



class TradingSim(object):
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps=1e-3, time_cost_bps=1e-4):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.mkt_nav = np.ones(self.steps)
        self.strat_retrns = np.ones(self.steps)
        self.posns = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.mkt_retrns = np.zeros(self.steps)

    def reset(self, train=True):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.mkt_nav.fill(1)
        self.strat_retrns.fill(0)
        self.posns.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.mkt_retrns.fill(0)

    def _step(self, action, retrn):
        bod_posn = 0.0 if self.step == 0 else self.posns[self.step - 1]
        bod_nav = 1.0 if self.step == 0 else self.navs[self.step - 1]
        mkt_nav = 1.0 if self.step == 0 else self.mkt_nav[self.step - 1]

        self.mkt_retrns[self.step] = retrn
        self.actions[self.step] = action

        self.posns[self.step] =  action  # was action - 1 for action in [0,1,2]
        self.trades[self.step] = self.posns[self.step] - bod_posn

        trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps
        self.costs[self.step] = trade_costs_pct + self.time_cost_bps
        reward = ((bod_posn * retrn) - self.costs[self.step]) # see if can change to sharp ind
        self.strat_retrns[self.step] = reward
        logging.debug(
            "debug ----- :retrn:%f,action:%d,bod_posn,%d,posn:%d,trades:%d,trade_costs_pct:%f,costs:%f,reward:%f" % (
                retrn, action,
                bod_posn,
                self.posns[self.step],
                self.trades[self.step],
                trade_costs_pct,
                self.costs[self.step],
                reward))

        if self.step != 0:
            self.navs[self.step] = bod_nav * (1 + self.strat_retrns[self.step - 1])
            self.mkt_nav[self.step] = mkt_nav * (1 + self.mkt_retrns[self.step - 1])

        info = {'reward': reward, 'nav': self.navs[self.step], 'costs': self.costs[self.step],
                'pos': self.posns[self.step]}
        self.step += 1
        return reward, info

    def to_df(self):
        """returns internal state in new dataframe """
        cols = ['action', 'bod_nav', 'mkt_nav', 'mkt_return', 'sim_return',
                'position', 'costs', 'trade']

        df = pd.DataFrame({'action': self.actions,  # today's action (from agent)
                           'bod_nav': self.navs,  # BOD Net Asset Value (NAV)
                           'mkt_nav': self.mkt_nav,  #
                           'mkt_return': self.mkt_retrns,
                           'sim_return': self.strat_retrns,
                           'position': self.posns,  # EOD position
                           'costs': self.costs,  # eod costs
                           'trade': self.trades},  # eod trade
                          columns=cols)
        return df


class TradingEnv(object):
    """This gym implements a simple trading environment for reinforcement learning.
    The gym provides daily observations based on real market data pulled
    from Quandl on, by default, the SPY etf. An episode is defined as 252
    contiguous days sampled from the overall dataset. Each day is one
    'step' within the gym and for each step, the algo has a choice:
    SHORT (0)
    FLAT (1)
    LONG (2)
    If you trade, you will be charged, by default, 10 BPS of the size of
    your trade. Thus, going from short to long costs twice as much as
    going from short to/from flat. Not trading also has a default cost of
    1 BPS per step. Nobody said it would be easy!
    At the beginning of your episode, you are allocated 1 unit of
    cash. This is your starting Net Asset Value (NAV). If your NAV drops
    to 0, your episode is over and you lose. If your NAV hits 2.0, then
    you win.
    The trading envs will track a buy-and-hold strategy which will act as
    the benchmark for the game.
    """

    def __init__(self, symbol, interval, start, end, freq_cnt=1440, random = True):
        self.freq_cnt = freq_cnt
        self.src = ZiplineEnvSrc(symbol=symbol, interval=interval, start=start, end=end, freq_cnt=self.freq_cnt)
        self.sim = TradingSim(steps=self.freq_cnt, trading_cost_bps=1e-4, time_cost_bps=1e-4)  # TODO FIX

        self.action_space = [0,1] # was (0,1,2), 0 is short, for now, only consider 1,2
        self.observation_space = self.src.data.shape[1]
        
        self.render_on = 0
        self.reset_count = 0
        self.random = random
        self.reset()
        

    def _step(self):
        observation, done = self.src._step()
        return observation, done
        
    def _rewards_ls(self, observation, action_ls):
        reward_ls = []
        self.src.step -= self.freq_cnt
        for i,act in enumerate(action_ls):
            yret = observation[i+1][0] #return
            reward, info = self.sim._step(act, yret)
            reward_ls.append(reward)
        return reward_ls

    def reset(self):
        self.reset_count += 1
        self.src.reset(self.random)
        self.sim.reset()
        return self.src._step()[0]
