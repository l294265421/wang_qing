from data.gen_analysis_data import *

gov_vehicle_alarm_monthly = gov_vehicle_alarm_monthly[gov_vehicle_alarm_monthly['StatMonth'].map(lambda x : x != 197001 and x != 197002)]

col = ['VehicleID', 'StatMonth', 'Exigency']
exigency_data = gov_vehicle_alarm_monthly[col]
exigency_data = pd.pivot_table(data=exigency_data, columns='StatMonth', index='VehicleID', values='Exigency').fillna(0)
exigency_data_path = data_dir + 'exigency_data.csv'
exigency_data.to_csv(exigency_data_path, sep=' ', header=False, index=False)