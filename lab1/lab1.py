import pandas as pd
import re
excel = pd.read_excel('data.xlsx')

df = excel.head(10)
clean_frame = pd.DataFrame(columns=['Tag', 'Latitude', 'Longitude'])
index = 0

for index, row in df.iterrows():
    tags = re.findall(r"#(\w+)", row['Tweet content'])
    latidute = row['Latitude']
    longitude = row['Longitude']
    print(tags)
    for tag in tags:
        print(tag)
        print(index)
        clean_frame.loc[index] = [tag, latidute, longitude]
        index += 1