import os

import pandas as pd


def find_checkins(id):
    df = pd.read_csv(os.path.join("data", "biz_csv", 'business.csv'))
    print(df)


find_checkins(124296)
