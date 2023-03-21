import train_rf_model
import pandas as pd
import joblib

symbol_file = '/opt/demos/SampleStocks.csv'
symbol = pd.read_csv(symbol_file, index_col=None)['Code'].to_list()

if __name__ == "__main__":
    for i in range(len(symbol)):
        my_model = train_rf_model.Adjust_Stock_Delta(symbol[i])
        joblib.dump(my_model, 'models/{}.pkl'.format(symbol[i]))
