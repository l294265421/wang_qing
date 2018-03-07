from data.raw_data import *

invalid_month = [197001, 197002,201412, 201501, 201502, 201607]
gov_vehicle_alarm_monthly = gov_vehicle_alarm_monthly[gov_vehicle_alarm_monthly['StatMonth'].map(lambda x : x not in invalid_month)]
# 超速驾驶
overspeed_columns = ['Overspeed2_5', 'Overspeed2_510', 'Overspeed2_10',
       'Overspeed25_510', 'Overspeed25_10', 'Overspeed25_5', 'Overspeed5_5',
       'Overspeed5_510', 'Overspeed5_10', 'NightOverspeed2_5',
       'NightOverspeed2_510', 'NightOverspeed2_10', 'NightOverspeed25_5',
       'NightOverspeed25_510', 'NightOverspeed25_10', 'NightOverspeed5_5',
       'NightOverspeed5_510', 'NightOverspeed5_10']
gov_vehicle_alarm_monthly['overspeed_all'] = 0
for column in overspeed_columns:
    gov_vehicle_alarm_monthly['overspeed_all'] = gov_vehicle_alarm_monthly['overspeed_all'] + gov_vehicle_alarm_monthly[column]

need_columns = ['VehicleID','StatMonth','Exigency','overspeed_all','FatigueDrive','VehicleType','ZoneID','CompanyID']
gov_vehicle_alarm_monthly = gov_vehicle_alarm_monthly[need_columns]

# gov_vehicle_alarm_monthly = pd.get_dummies(gov_vehicle_alarm_monthly,prefix='vehicle_type', columns=['VehicleType'])
#
# gov_vehicle_alarm_monthly = pd.get_dummies(gov_vehicle_alarm_monthly, prefix='zone_id', columns=['ZoneID'])
#
# gov_vehicle_alarm_monthly = pd.get_dummies(gov_vehicle_alarm_monthly,prefix='company_id', columns=['CompanyID'])

gov_vehicle_alarm_monthly.sort_values(['VehicleID','StatMonth'], axis=0).to_csv(data_dir + 'gov_vehicle_alarm_monthly_need_data.csv', index=False)

# 从201606开始往前统计，数量少于8的用户被删除掉
months = [201503, 201504, 201505, 201506, 201507, 201508, 201509, 201510, 201511, 201512, 201601, 201602,
          201603, 201604, 201605, 201606]

gov_vehicle_alarm_monthly_need_data_normal = open(data_dir + 'gov_vehicle_alarm_monthly_need_data_normal.csv', mode='w')

def deal_one_user_records(single_user_records):
    valid_single_user_recoreds = []
    for i in range(-1, -1 * len(single_user_records), -1):
        if str(months[i]) == single_user_records[i][1]:
            valid_single_user_recoreds.append(','.join(single_user_records[i]))
        else:
            break
    if len(valid_single_user_recoreds) >= 8:
        valid_single_user_recoreds.reverse()
        gov_vehicle_alarm_monthly_need_data_normal.write(';'.join(valid_single_user_recoreds) + '\n')

with open(data_dir + 'gov_vehicle_alarm_monthly_need_data.csv') as gov_vehicle_alarm_monthly_need_data_file:
    single_user_records = []
    current_user = ''
    gov_vehicle_alarm_monthly_need_data_file.readline()
    lines = gov_vehicle_alarm_monthly_need_data_file.readlines()
    count = 0
    for line in lines:
        count += 1
        parts = line.strip().split(',')
        if current_user == '':
            current_user = parts[0]
            single_user_records.append(parts)
        else:
            if current_user == parts[0]:
                single_user_records.append(parts)
                if count == len(lines):
                    deal_one_user_records(single_user_records)

            else:
                deal_one_user_records(single_user_records)
                current_user = parts[0]
                single_user_records = [parts]


gov_vehicle_alarm_monthly_need_data_normal.close()





