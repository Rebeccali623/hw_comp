import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import interact, fixed

symbol_file = '/opt/demos/SampleStocks.csv'
symbol = pd.read_csv(symbol_file, index_col=None)['Code'].to_list()

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.set_option('display.max_rows', 7)
pd.set_option('display.max_columns', 9)


def df_interact(df, nrows=7, ncols=7):
    def peek(row=0, col=0):
        return df.iloc[row:row + nrows, col:col + ncols]

    row_arg = (0, len(df), nrows) if len(df) > nrows else fixed(0)
    col_arg = ((0, len(df.columns), ncols)
               if len(df.columns) > ncols else fixed(0))

    interact(peek, row=row_arg, col=col_arg)
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))

tickdata_0306 = pd.read_csv('/opt/new_tickdata/tickdata_20230306.csv')
tickdata_0307 = pd.read_csv('/opt/new_tickdata/tickdata_20230307.csv')
tickdata_0308 = pd.read_csv('/opt/new_tickdata/tickdata_20230308.csv')
tickdata_0309 = pd.read_csv('/opt/new_tickdata/tickdata_20230309.csv')
tickdata_0310 = pd.read_csv('/opt/new_tickdata/tickdata_20230310.csv')
tickdata_0313 = pd.read_csv('/opt/new_tickdata/tickdata_20230313.csv')
tickdata_0314 = pd.read_csv('/opt/new_tickdata/tickdata_20230314.csv')
tickdata_0315 = pd.read_csv('/opt/new_tickdata/tickdata_20230315.csv')
tickdata_0316 = pd.read_csv('/opt/new_tickdata/tickdata_20230316.csv')
tickdata = pd.concat([tickdata_0306, tickdata_0307, tickdata_0308, tickdata_0309,
                      tickdata_0310, tickdata_0313, tickdata_0314, tickdata_0315], axis=0).reset_index(drop=True)
columns_name = ['Index', 'Stock code', 'Tick time', 'Open', 'High', 'Low', 'Last transaction price'] + (
    ['Selling Price' + ' ' + str(i) for i in range(1, 11)]) + (
                   ['Selling Volume' + ' ' + str(i) for i in range(1, 11)]) + (
                   ['Buying Price' + ' ' + str(i) for i in range(1, 11)]) + (
                   ['Buying Volume' + ' ' + str(i) for i in range(1, 11)])
train_data = tickdata.iloc[:, :47]
train_data.columns = columns_name
# Stock_code = train_data['Stock code'].unique().tolist()


def tmp_stock_code(code, data=train_data):
    tmp = data[data['Stock code'] == code]
    return tmp


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


def labelling(df, delta_t, class_interval, plot=False):
    df_tmp = df.assign(
        future_return=(df['Last transaction price'].shift(-delta_t) - df['Last transaction price']) / df[
            'Last transaction price']
    ).dropna()
    bins = [df_tmp['future_return'].min()]
    for interval in class_interval:
        bins = bins + [df_tmp['future_return'].quantile(interval)]
    bins = bins + [df_tmp['future_return'].max()]
    labels = list(range(len(class_interval) + 1))
    df_after_label = df_tmp.assign(
        label=lambda x: pd.cut(x['future_return'], bins=bins, labels=labels, include_lowest=True))
    return df_after_label


tmp_600536 = tmp_stock_code('600536.SH')
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier


def keep_features(X, features):
    return X.loc[:, features]


def mini_model(features, model):
    return make_pipeline(
        FunctionTransformer(keep_features, kw_args=dict(features=features)),
        model,
    )


tmp_tmp = all_feature(tmp_600536)
features = tmp_tmp.loc[:, 'Selling Price 1':'Buying Volume 10_Diff'].columns.tolist()
basic_set = features[0:40]
time_insensitive_set = features[40:-40]
time_sensitive_set = features[-40:]
mid_price = tmp_tmp.filter(like='Mid Price', axis=1).columns.tolist() + ['VWAP', 'Mean_Price_Ask', 'Mean_Price_Bid']
mid_price_time_sensitive = mid_price + tmp_tmp.filter(regex='Diff$', axis=1).filter(like='Price',
                                                                                    axis=1).columns.tolist()
price_spread = tmp_tmp.filter(like='Price Spread', axis=1).columns.tolist() + ['VWAP']
volume_spread = tmp_tmp.filter(like='Volume Spread', axis=1).columns.tolist() + ['VWAP']
best_ask_bid = tmp_tmp.filter(regex='1$', axis=1).columns.tolist() + ['VWAP'] + tmp_tmp.filter(regex='1_Diff$',
                                                                                               axis=1).columns.tolist()

### TEST DATA SAMPLE(根据实际的测试集修改）
test_data = tickdata_0316.iloc[:, :47]
test_data.columns = columns_name
### MODEL
rf = RandomForestClassifier(n_estimators=200, max_depth=100, max_features='sqrt',
                            criterion='entropy', n_jobs=4)


def Adjust_Stock_Delta(stock, delta=10, interval=[0.2, 0.8]):
    tmp_stock = tmp_stock_code(stock)
    data = labelling(all_feature(tmp_stock), delta, interval)
    X_train = data.loc[:, 'Selling Price 1':'Buying Volume 10_Diff']
    y_train = data['label']
    print("Training Data Size: ", len(X_train))
    rf_time_sensitive = mini_model(time_insensitive_set, rf).fit(X_train, y_train)
    return rf_time_sensitive


# models = []
# for stock in Stock_code:
#     models.append(Adjust_Stock_Delta(stock))
# models_dict = pd.DataFrame({
#     'stock_id': Stock_code, 'model': models
# })
# models_dict.set_index('stock_id')

