import pandas as pd
import os
import sys

data_dir = r'D:\document\program\ml\machine-learning-databases\wangqing\\'
origional_data_dir = r'G:\partofdata\\'

def load_data(filename, nrows):
    if not os.path.exists(data_dir + filename):
        pd.read_csv(origional_data_dir + filename, nrows=nrows).to_csv(data_dir + filename, index=False)
    return pd.read_csv(data_dir + filename)


gov_vehicle_alarm_monthly = load_data('GOVVehicleAlarmMonthly.csv', sys.maxsize)

# pd.read_csv(origional_data_dir + 'GOV6BanVehicleDaily.csv', nrows=10)
#
# pd.read_csv(origional_data_dir + 'GOVExigencyDetail.csv', nrows=10)
#
# pd.read_csv(origional_data_dir + 'GOVFatigueDriveDetail.csv', nrows=10)
#
# pd.read_csv(origional_data_dir + 'GOVOverspeedDetail.csv', nrows=10)

