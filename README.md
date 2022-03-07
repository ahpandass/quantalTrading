# quantalTrading

It is coding for quantal trading.
Mainly 2 part, 
1. TriangleProfit is looking for 3 assets who has price gap, and quick trade 3 time for tiny benifit.
2. RL trading is trying to trade RL models to learn how to do trading.

problem I met so far:
for TraiangleProfit, there are too many rubbish assets disturb the oberservation, thos assets normally have very low order frequency and big gap in orderbook. In that case, when we tracking the profit, it will be easily misleading by some noise orders.

and for RL trading, the model is not stable, even involve the Talib and try to use sharp ind as reward function, it still not perform well.
and when try to use CNN, it is not even convergence, not sure if the structure of the feature dataframe not feeding the model or the model is too simple, need more investigate.
