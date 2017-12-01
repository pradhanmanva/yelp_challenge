import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixing random state for reproducibility
np.random.seed(19656801)

business_df = pd.read_csv(os.path.join("data", "biz_csv", 'business.csv'))
print(business_df)
N = 20
y = business_df['longitude'].values
x = business_df['latitude'].values
col = np.random.rand(len(x))
area = np.pi * (15 * np.random.rand(N)) ** 2  # 0 to 15 point radii

plt.scatter(x, y, c=col)
plt.show()
