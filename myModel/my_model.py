import sys
import pandas as pd
import joblib

input_path = sys.argv[1]
output_path = sys.argv[2]
symbol_file = '/opt/demos/SampleStocks.csv'

tick_data = open(input_path, 'r')
order_time = open(output_path, 'w')
symbol = pd.read_csv(symbol_file, index_col=None)['Code'].to_list()
idx_dict = dict(zip(symbol, list(range(len(symbol)))))

models_dict = {}
for i in range(len(symbol)):
    model_file = 'models/{}.pkl'.format(symbol[i])
    model = joblib.load(model_file)
    models_dict.update({symbol[i]:model})
    print(i)

columns_name = ['Index', 'Stock code', 'Tick time', 'Open', 'High', 'Low', 'Last transaction price'] + (
    ['Selling Price' + ' ' + str(i) for i in range(1, 11)]) + (
                   ['Selling Volume' + ' ' + str(i) for i in range(1, 11)]) + (
                   ['Buying Price' + ' ' + str(i) for i in range(1, 11)]) + (
                   ['Buying Volume' + ' ' + str(i) for i in range(1, 11)])


def all_feature(df):
    tmp = df.copy()
    for i in range(1, 11):
        tmp['Mid Price' + str(i)] = (tmp['Selling Price' + ' ' + str(i)] + tmp['Buying Price' + ' ' + str(i)]) / 2
        tmp['Price Spread' + str(i)] = (tmp['Selling Price' + ' ' + str(i)] - tmp['Buying Price' + ' ' + str(i)])
        tmp['Volume Spread' + str(i)] = (tmp['Selling Volume' + ' ' + str(i)] - tmp['Buying Volume' + ' ' + str(i)])
    tmp = tmp.assign(
        VWAP=lambda x: (
            x['Selling Price 1']*x['Buying Volume 1']+x['Buying Price 1']*x['Selling Volume 1'])/(x['Selling Volume 1']+x['Buying Volume 1'])
          ).assign(
        Mean_Price_Ask=tmp.filter(like='Selling Price', axis=1).mean(axis=1)
    ).assign(
        Mean_Price_Bid=tmp.filter(like='Buying Price', axis=1).mean(axis=1)
    ).assign(
        Mean_Volume_Ask=tmp.filter(like='Selling Volume', axis=1).mean(axis=1)
    ).assign(
        Mean_Volume_Bid=tmp.filter(like='Buying Volume', axis=1).mean(axis=1)
    ).assign(
        Accumulated_Spread=tmp.filter(like='Price Spread', axis=1).sum(axis=1)
    ).assign(
        Accumulated_Volume_Spread=tmp.filter(like='Volume Spread', axis=1).sum(axis=1)
    )
    tmp_time_sensitive = tmp.loc[:, 'Selling Price 1':'Buying Volume 10'] - tmp.loc[:, 'Selling Price 1':'Buying Volume 10'].shift(1)
    tmp = tmp.join(tmp_time_sensitive, rsuffix='_Diff').reset_index(drop=True)
    return tmp.loc[1:, :]


tick_dic = {}  # to record previous tick for this symbol
for i in range(len(symbol)):
    tick_dic.update({symbol[i]:[]})

target_vol = 100
basic_vol = 2
cum_vol_buy = [0] * len(symbol)  # accumulate buying volume
cum_vol_sell = [0] * len(symbol)  # accumulate selling volume

header = tick_data.readline()  # header
header = header[:-1].split(',')[:47]
order_time.writelines('symbol,BSflag,dataIdx,volume\n')
order_time.flush()
i = -1
while True:
    i+=1
    if i % 10000 == 0:
        print(i)
    row = tick_data.readline()  # read one tick line
    if row.strip() == 'stop' or len(row) == 0:
        break
    row = row[:-1].split(',')
    sym = row[1]
    idx = idx_dict[sym]

    row[1] = '-1'
    row = list(map(int, row))[:47]
    row[1] = sym

    order = ('N', 0)

    if cum_vol_buy[idx] < target_vol or cum_vol_sell[idx] < target_vol:
        if tick_dic[sym] == []:
            tick_dic[sym] = row
        else:
            prev_tick = tick_dic[sym]
            df = pd.DataFrame([prev_tick, row], columns=header)
            df.columns = columns_name
            df = all_feature(df)

            label = models_dict[sym].predict(df)[0]
            if label == 2 and cum_vol_buy[idx] < target_vol:
                order = ('B', 1)
                cum_vol_buy[idx] += 1
            elif label == 0 and cum_vol_sell[idx] < target_vol:
                order = ('S', 1)
                cum_vol_sell[idx] += 1

    if order[0] == 'N':
        order_time.writelines(f'{sym},N,{i},0\n')
        order_time.flush()
    else:
        order_time.writelines(f'{sym},{order[0]},{i},{order[1]}\n')
        order_time.flush()


tick_data.close()
order_time.close()