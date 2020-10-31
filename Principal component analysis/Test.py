import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import statistics
import pandas as pd
pd.set_option('precision', 3)
pd.set_option('display.max_columns', 100)


results = [0.862443, 0.890267,0.886055,0.062145,0.107230,0.140876]
df = pd.DataFrame(results)
def variance (df):
    # Общее среднее значение
    totalAVGSum = 0
    totalSum = 0
    for i in range(len(df.columns)):
        totalAVGSum = totalAVGSum + df[i].mean()
        totalSum = totalSum = np.sum(df[i])

    totalResVar = 0
    totalOutgroupp = 0
    totalIngroupp = 0
    for i in range(len(df.columns)):
        # Внутригрупповая дисперсия
        totalIngroupp = 0
        for j in range(len(df[i])):
            totalIngroupp = totalIngroupp + np.power(j-df[i].mean(), 2)*df[i][j]/np.sum(df[i])
        print("Внутригрупповая дисперсия группы ", i, ": ", totalIngroupp)

        # Остаточная дисперсия
        totalResVar = totalResVar + (totalIngroupp * np.sum(df[i]))/totalSum

        # Межгрупповая дисперсия
        totalOutgroupp = totalOutgroupp + np.power(df[i].mean()-totalAVGSum, 2)*np.sum(df[i])/np.sum(df[i])

    print("Остаточная дисперсия: ", totalResVar)
    print("Межгрупповая дисперсия ", i, ": ", totalOutgroupp)
    # Общая дисперсия
    print("Общая дисперсия: ", i, ": ", totalResVar + totalOutgroupp)



