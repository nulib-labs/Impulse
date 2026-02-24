import pandas as pd

df = pd.read_csv("emissions/emissions.csv")
print(df.columns)
print(len(df))
print(sum(df.emissions))
print(sum(df.energy_consumed))
print(sum(df.ram_used_gb))
print(sum(df.energy_consumed))
print(sum(df.water_consumed))
