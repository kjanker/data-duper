import pandas as pd
# load data from https://data.open-power-system-data.org/renewable_power_plants/
data_file = "https://data.open-power-system-data.org/renewable_power_plants/2020-08-25/renewable_power_plants_SE.csv"
df_real = pd.read_csv(data_file, parse_dates=['commissioning_date'])

from duper import Duper
duper = Duper()
duper.fit(df=df_real)
df_dupe = duper.make(size=10000)

print(duper)