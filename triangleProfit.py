#tail -fn 50 log.out
#kill -9 1367
#ps -aux | grep python3
#nohup python3 triangleprofit.py > log.out 2>&1 &
import requests,time, concurrent.futures
from datetime import datetime
import pandas as pd
import ccxt

binance_exchange = ccxt.binance({
    'apiKey': '9tKYUI74alkCHK1p7uEaBqvxFJfCYnNANsFfLY653fje4IdBejoB7UAg32pw4JqS',
    'secret': 'H5oEJ2X7qn0nJgk1ySkGeksVphyWmgCdb3djeP7XUxHzb3xqulASOqn4yp5nXeiu',
    'timeout': 15000,
    'enableRateLimit': True,
    'options': { 'adjustForTimeDifference': True }
    })

market_a = 'USDT'
market_b = 'BNB'

def write_log_file(log_msg):
    destFile = r"log-{}.txt".format(session_start_datetime)
    with open(destFile, 'a') as f:
        f.write(log_msg+'\r\n')

def get_target_c(biance_markets,market_a='BTC',market_b='ETH'):
    symbols = biance_markets.keys()
    symbols_df = pd.DataFrame(symbols,columns=['symbol'])
    base_quote_df = symbols_df['symbol'].str.split('/',expand=True)
    base_quote_df.columns=['base','quote']
    quote_a_list = base_quote_df[base_quote_df['quote']==market_a]['base'].values.tolist()
    quote_b_list = base_quote_df[base_quote_df['quote']==market_b]['base'].values.tolist()
    common_quote_list = list(set(quote_a_list).intersection(set(quote_b_list)))
    return common_quote_list
    
        

def get_symbols(base,market_a='BTC',market_b='ETH'):
    market_c = base
    market_a2b_symbol = '{}/{}'.format(market_b,market_a)
    market_b2c_symbol = '{}/{}'.format(market_c,market_b)
    market_a2c_symbol = '{}/{}'.format(market_c,market_a)
    symbol_ls = (market_a2b_symbol,market_b2c_symbol,market_a2c_symbol)
    return symbol_ls

def get_ticker(base,market_a='BTC',market_b='ETH'):
    #print(datetime.now())
    time.sleep(binance_exchange.rateLimit / 1000)
    market_a2b_symbol,market_b2c_symbol,market_a2c_symbol = get_symbols(base,market_a,market_b)
    symbol_ls = (market_a2b_symbol,market_b2c_symbol,market_a2c_symbol)
    ticker_data_ls = binance_exchange.fetchTickers(symbol_ls)
    p1 = ticker_data_ls[market_a2b_symbol]['close']
    t1 = ticker_data_ls[market_a2b_symbol]['datetime']
    p2 = ticker_data_ls[market_b2c_symbol]['close']
    t2 = ticker_data_ls[market_a2b_symbol]['datetime']
    p3 = ticker_data_ls[market_a2c_symbol]['close']
    t3 = ticker_data_ls[market_a2b_symbol]['datetime']
    t = '/'.join([str(t1),str(t2),str(t3)])
    #print(datetime.now())
    if (p2>1e-4) and (p3>1e-4):
        profit = p3/(p2*p1)-1
        #print('end process: {},{},{},{},{}'.format(p1,p2,p3,t,profit))
        return [base,t,p1,p2,p3,profit]
    else:
        #print('price is not valid')
        return False

def buy_order(symbol,price,balance):
    #print('     start of buy order: {}:{}:{}'.format(symbol,price,balance))
    cnt = 0
    set_order = []
    while cnt<2:
        if balance > 0.002:
            #amt = balance/binance_exchange.fetchTickers(symbol)[symbol]['close']
            amt = balance/price
            if amt > 0.0:
                try:
                    set_order = binance_exchange.create_order(symbol=symbol, side='buy', type='limit',price = price,amount = amt)
                except Exception as ex:
                    print(ex)
                break
            else:
                return('Failure')
        else:
            time.sleep(binance_exchange.rateLimit / 1000)
            cnt += 1
    if set_order:
        return [set_order['info']['orderId'],symbol]
    else:
        return('Failure')

def sell_order(symbol,price=0,balance=0,isFailureSoldOut=False):
    #print('start of sell order: {}:{}'.format(symbol,price))
    cnt = 0
    set_order = []
    while cnt<2:
        if isFailureSoldOut and balance > 0:
            base_balance = balance
        else:
            base_balance = binance_exchange.fetch_balance()['free'][symbol.split('/')[0]]
        if base_balance > 0.005:
            set_order = binance_exchange.create_order(symbol=symbol, side='sell', type='market',amount = base_balance)
            break
        else:
            time.sleep(binance_exchange.rateLimit / 1000)
            cnt += 1
    if set_order:
        return [set_order['info']['orderId'],symbol]
    else:
        return('Failure')

def push_orders(symbol, price, balance, side):
    if side == 'buy':
        return buy_order(symbol,price,balance)
    if side == 'sell':
        return sell_order(symbol,price)
    return null


def start_multi_orders(symbol_price_ls):
    future_to_orders = []
    order_id_ls = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    # Start the load operations and mark each future with its symbol
        for symbol,price,balance, side in symbol_price_ls:
            future_to_orders.append(executor.submit(push_orders, symbol, price, balance, side))
        for future in concurrent.futures.as_completed(future_to_orders):
            try:
                order_id = future.result()
            except Exception as exc:
                return ['Failure']
            else:
                order_id_ls.append(order_id)
    return order_id_ls

def failure_sold_out(order_id_ls):
    orders_id_ls = []
    for order in order_id_ls:
        if order == 'Failure':
            continue
        else:
            order_details = get_order_info([order])
            if (order_details[0][4]=='buy') and (float(order_details[0][3])>0):
                print("     retreating the order: {}".format(order_details))
                orders_id_ls.append( sell_order(order_details[0][0],order_details[0][2],float(order_details[0][3]),isFailureSoldOut=True) )
    return orders_id_ls

def get_order_info(order_id_ls):
    order_details = []
    for order in order_id_ls:
        symbol = order[1]
        order_id = order[0]
        order_detail = binance_exchange.fetchOrder(order_id,symbol)
        date_time = order_detail['datetime']
        price = order_detail['price']
        quantity = order_detail['info']['executedQty']
        side = order_detail['side']
        order_details.append([symbol,date_time,price,quantity,side])
    return order_details


if __name__ == '__main__':
    print("Start initialization, and load market")
    biance_markets = binance_exchange.load_markets()
    common_quote_list = get_target_c(biance_markets,market_a,market_b)
    print(" Getting target coins")
    #common_quote_short_list=['ROSE', 'GNO', 'VOXEL', 'FIDA', 'ATA', 'ETC', 'ARPA', 'DYDX', 'FET', 'XMR', 'BAKE', 'XVS', 'BNX', 'GXS', 'MC', 'LRC', 'COCOS', 'CELR', 'ZEN', 'LINK', 'ALICE', 'XTZ', 'NEAR', 'IDEX', 'LOKA', 'THETA', 'MLN', 'CHZ', 'ENJ', 'TRX', 'FLOW', 'JASMY', 'MBOX', 'LPT', 'XLM', 'HARD', 'ERN', 'BURGER', 'DAR', 'SUSHI', 'HIGH', 'TROY', 'SXP', 'PAXG', 'PEOPLE', 'RUNE', 'MITH', 'FOR', 'IOTA', 'AGLD', 'FARM', 'SPELL', 'SOL', 'SAND', 'CAKE', 'AAVE', 'FIO', 'OGN', 'STX', 'QNT', 'WAVES', 'RARE', 'NEO', 'FIL', 'HBAR', 'AXS', 'CLV', 'LINA', 'COS', 'MANA', 'MBL', 'DASH', 'RAY', 'ADA', 'XRP', 'CHR', 'VET', 'WAXP', 'ENS', 'DOT', 'CITY', 'EOS', 'YGG', 'AVA', 'AR', 'PERL', 'ATOM', 'ALGO', 'COTI', 'UNI', 'YFII', 'KLAY', 'SNX', 'MATIC', 'ANT', 'AVAX', 'C98', 'LUNA', 'WRX', 'RAD', 'PLA', 'ANKR', 'GALA', 'AMP', 'SLP', 'TORN',  'ILV', 'OCEAN', 'ZEC', 'ICP', 'FTM', 'SRM', 'BEL', 'INJ', 'CTK', 'OOKI', 'BETA', 'RGT', 'FTT', 'MOVR', 'ONE', 'KSM', 'MASK', 'POLS', 'KAVA', 'EGLD', 'MINA', 'LTC', 'WOO', 'CTSI', 'ALPHA', 'TRIBE', 'BCH']
    common_quote_short_list = ['SOL']
    if len(common_quote_list)>len(common_quote_short_list):
        common_quote_list=common_quote_short_list
    print(" target coins has been filtered")
    session_start_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
    print(" Session start at {}".format(session_start_datetime))
    price_columns = ['c_symbol','timestamp','p1','p2','p3','profit']
    price_df = pd.DataFrame(columns=price_columns)
    order_columns = ['c_symbol','expect_profit','actual_profit']
    order_df = pd.DataFrame(columns=order_columns)
    balance_ls = binance_exchange.fetch_balance()['free']
    pre_balance = balance = balance_ls[market_a]
    balance_b_realtime = balance_ls[market_b]
    balance_b = 0.055
    balance_b_delta = 0.0
    print('initialization finished, with {} balance: {}'.format(market_a,balance))
    
    
    while True:
        for (i,base) in enumerate(common_quote_list):
            market_a2b_symbol,market_b2c_symbol,market_a2c_symbol = get_symbols(base,market_a,market_b)
            
            balance_b_realtime_before_txn = balance_ls[market_b]
            
            symbol_price_ls = []
            ticker_start_datetime = datetime.now().strftime('%y-%m-%d:%H%M%S')
            ticker = get_ticker(base,market_a,market_b)
            if not ticker:
                common_quote_list.remove(base)
                continue
            print('checking: {}: {} profit: {}'.format(i, base,ticker[5] ))
            if ticker:
                if ticker[5]>0.005:
                    print('ticker start at: ticker_start_datetime: {}'.format(ticker_start_datetime))
                    symbol_price_ls = [[market_a2b_symbol,ticker[2],(balance_b+balance_b_delta)*ticker[2],'buy'],[market_b2c_symbol,ticker[3],balance_b,'buy'],[market_a2c_symbol,ticker[4],balance_b,'sell']]
                    order_id_ls = start_multi_orders(symbol_price_ls)
                    
                    if 'Failure' in order_id_ls:
                        print("Failed orders: {}".format(order_id_ls))
                        order_details = failure_sold_out(order_id_ls)
                        print('transaction failed and removed from list')
                        common_quote_list.remove(base)
                        print('sleep 2 sec after transaction')
                        time.sleep(2)
                        continue
                    
                    order_details = get_order_info(order_id_ls)
                    for order_detail in  order_details:
                        if order_detail[0] == market_a2b_symbol:
                            p1_info = order_detail
                        if order_detail[0] == market_b2c_symbol:
                            p2_info = order_detail
                        if order_detail[0] == market_a2c_symbol:
                            p3_info = order_detail
                    
                    print('----------expect to buy {}, actual buy: {}'.format(balance_b+balance_b_delta, p1_info[3]))
                    balance_ls = binance_exchange.fetch_balance()['free']
                    balance = balance_ls[market_a]
                    balance_b_realtime_after_txn = balance_ls[market_b]
                    balance_b_delta= balance_b_realtime_before_txn - balance_b_realtime_after_txn
                    
                    print('     {} cost {}'.format(market_b, balance_b_delta))
                    
                    profit = (balance  - pre_balance) / pre_balance
                    now_time = datetime.now().strftime("%y-%m-%d:%H-%M-%S")
                    txn_log_msg='transaction time: {}, base coin: {},expect profit: {}, actual profit: {}, base_balance: {}, quote_balance: {}'.format(now_time,base,ticker[5],profit, balance, balance_b)
                    txn_detail_log_msg='        buy p1 at: {}, buy p2 at: {}, buy p3 at: {} '.format(p1_info[1],p2_info[1],p3_info[1])
                    txn_detail_log_price_msg='      profit: {}/{}, p1: {}/{}, p2: {}/{}, p3: {}/{} '.format((p3_info[2]/(p1_info[2]*p2_info[2])-1),ticker[5], p1_info[2],ticker[2],p2_info[2],ticker[3],p3_info[2],ticker[4])
                    print(txn_log_msg)
                    print(txn_detail_log_msg)
                    print(txn_detail_log_price_msg)
                    write_log_file(txn_log_msg)
                    write_log_file(txn_detail_log_msg)
                    write_log_file(txn_detail_log_price_msg)
                                        
                    pre_balance = balance 
                    rate = 1
                    if balance_b*p1_info[2] > balance:
                        break
                    print('sleep 2 sec after transaction')
                    time.sleep(2)
            else:
                common_quote_list.remove(base)
            if (i%30)==0:
                time.sleep(2)
                print(' 2 second wait after 30 scan')
        print('All coins scanned, new started')

