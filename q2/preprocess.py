import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv("data/origin/表2-患者影像信息血肿及水肿的体积及位置.csv")
df_time = pd.read_csv("data/origin/time.csv")


print(df_time.head())
print(df_time.info())


