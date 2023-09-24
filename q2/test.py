from datetime import datetime,timedelta

time = datetime.strptime('2018/7/18  17:53', '%Y/%m/%d %H:%M') - datetime.strptime('2018/1/20 9:18', '%Y/%m/%d %H:%M')
print(time.total_seconds()/3600+3)
