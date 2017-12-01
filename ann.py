import os

import pandas as pd


def find_checkins(id):
    df = pd.read_csv(os.path.join("data", "checkin_csv", 'checkin.csv'))
    print(df.loc[df["business_id"] == id].sort_values(['no_of_checkins']))

find_checkins(124296)
