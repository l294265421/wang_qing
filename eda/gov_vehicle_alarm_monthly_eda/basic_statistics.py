from data.gen_analysis_data import *

gov_vehicle_alarm_monthly.head(10)

print('行数：' + str(len(gov_vehicle_alarm_monthly)))

gov_vehicle_alarm_monthly.columns

for column_name in gov_vehicle_alarm_monthly.columns:
    print('{} : {}'.format(column_name, len(gov_vehicle_alarm_monthly[column_name].unique())))

gov_vehicle_alarm_monthly['StatMonth'].unique()

len(gov_vehicle_alarm_monthly[gov_vehicle_alarm_monthly['StatMonth'].map(lambda x : x != 197001 and x != 197002)])
