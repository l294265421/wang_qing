from data.raw_data import data_dir

columns = ['VehicleID','StatMonth','Exigency','overspeed_all','FatigueDrive','VehicleType','ZoneID','CompanyID']

test = False
test_sample_num = 500

exigency_data = []
sample_count = 0
with open(data_dir + 'gov_vehicle_alarm_monthly_need_data_normal.csv') as data_normal_file:
    for line in data_normal_file:
        if test and sample_count >= test_sample_num:
            break
        one_sample = [[element for element in month.split(',')] for month in line.split(';')]
        one_sample = [[month[2]] + month[5:] for month in one_sample]
        exigency_data.append([[float(feature) for feature in month] for month in one_sample])
        sample_count += 1
