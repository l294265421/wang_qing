from data.raw_data import data_dir

columns = ['VehicleID','StatMonth','Exigency','overspeed_all','FatigueDrive','VehicleType','ZoneID','CompanyID']

fatigue_drive_data = []
with open(data_dir + 'gov_vehicle_alarm_monthly_need_data_normal.csv') as data_normal_file:
    for line in data_normal_file:
        one_sample = [[element for element in month.split(',')] for month in line.split(';')]
        one_sample = [[month[4]] + month[5:] for month in one_sample]
        fatigue_drive_data.append([[float(feature) for feature in month] for month in one_sample])
