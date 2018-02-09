"""
import pandas as pd
data = pd.read_csv("iris.data",names =["sepal length","sepal width","petal length","petal width","cat"])
print(data.head())
sepal_len_cnt = data['sepal length'].value_counts()
print(sepal_len_cnt)

#8.3.2 stockdata
import pandas as pd
import numpy as np
stockdata = pd.read_csv("dow_jones_index.data",parse_dates = ['date'],index_col =['date'],nrows = 100)
print(stockdata.head())
print(max(stockdata['volume']))
print(stockdata.index.day)
print(stockdata.resample('M').apply(np.sum))
stockdata.drop(["percent_change_volume_over_last_wk"], axis = 1)
stockdata_new = pd.DataFrame(stockdata, columns = ['stock', 'open', 'high', "low", "close", "volume"])
print(stockdata_new.head())

stockdata["previous_weeks_volume"] = 0

print(stockdata.dropna().head(2))
stockdata_new.open.describe() 
stockdata_new.open = pd.to_numeric(stockdata_new.open.str.replace('$', ''))
stockdata_new.close = pd.to_numeric(stockdata_new.close.str.replace('$', ''))
print(stockdata_new.open.describe())

stockdata_new['newopen'] = stockdata_new.open.apply(lambda x: 0.8*x)
print(stockdata_new.newopen.head(5))

stockAA = stockdata_new.query('stock=="AA"')
print(stockAA.head())

"""


# 8.4 matplotlib
import matplotlib.pyplot as plt
from matplotlib import figure
import pandas as pd
import numpy as np
#import matplotlib as mpl
#mpl.use("TkAgg") # Use TKAgg to show figures

stockdata = pd.read_csv("dow_jones_index.data",parse_dates = ['date'],index_col =['date'],nrows = 100)

stockdata_new = pd.DataFrame(stockdata, columns = ['stock', 'open', 'high', "low", "close", "volume"])

stockCSCO = stockdata_new.query('stock=="CSCO"')
print(stockCSCO.head())

plt.figure()
plt.scatter(stockdata_new.index.date, stockdata_new.volume)
plt.xlabel('day')
plt.ylabel('stock close value')
plt.title('title')
plt.show()
#plt.savefig("nltkplot01.png")

# 8.4.1 子图绘制
stockAA = stockdata_new.query('stock=="AA"')

plt.subplot(2, 2, 1)
plt.plot(stockAA.index.weekofyear, stockAA.open, 'r--')
plt.subplot(2, 2, 2)
plt.plot(stockCSCO.index.weekofyear, stockCSCO.open, 'g-*')
plt.subplot(2, 2, 3)
plt.plot(stockAA.index.weekofyear, stockAA.open, 'g--')
plt.subplot(2, 2, 4)
plt.plot(stockCSCO.index.weekofyear, stockCSCO.open, 'r-*')
plt.savefig('nltkplot02.png')

# bad test
x = [1, 3, 4, 5, 8, 14]
y = [0, 2, 4, 7, 9, 19]
fig, axes = plt.subplots(nrows = 1, ncols = 2)
for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')
#plt.show(ax)
plt.savefig("nltkplot03.png")

# 8.4.2 添加坐标轴
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y, 'r')
#plt.show(axes)
plt.savefig("nltkplot04.png")

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(stockAA.index.weekofyear, stockAA.open, label="AA")
ax.plot(stockAA.index.weekofyear, stockCSCO.open, label="CSCO")
ax.set_xlabel("weekofyear")
ax.set_ylabel("stock value")
ax.set_title('Weekly change in stock price')
ax.legend(loc = 2)
plt.savefig("nltkplot05.png")

# 8.4.3 绘制散点图
plt.scatter(stockAA.index.weekofyear, stockAA.open)
plt.savefig("nltkplot06.png")

# 8.4.4 绘制条形图
n = 12
X = np.arange(n)
Y1 = np.random.uniform(0.5, 1.0, n)
Y2 = np.random.uniform(0.5, 1.0, n)
plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
plt.savefig("nltkplot07.png")

# 8.4.5 3D绘图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2, Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
#plt.show(ax)
plt.savefig("nltkplot08.png")

