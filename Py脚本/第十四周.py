# %% [markdown]
# 1. 使用 Pandas 网络数据阅读器从雅虎财经获取金融数据
#

# %%
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分
from sklearn.pipeline import make_pipeline
# KNeighborsRegressor ————> k近邻回归
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression  # LinearRegression ————> 线性回归
import numpy as np
import math
from sklearn.preprocessing import scale
import pandas_ta  # 这是一个面向股市的库,下方有更详细的介绍。
import matplotlib.pyplot as plt
from matplotlib import style  # 绘图样式
import matplotlib as mpl
import pandas as pd  # 网络数据阅读器
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
start = datetime.datetime(2021, 5, 27)  # 开始时间
end = datetime.datetime(2022, 5, 26)  # 结束时间
# 从雅虎获取数据，雅虎已经停止向大陆提供服务，需要连接代理使用该方法。
df = web.DataReader("TSLA", 'yahoo', start, end)


# %%
df.tail()  # tail()方法一般用来对数据集进行矩阵形式的显示，默认显示为数据集的最后5行。


# %% [markdown]
# 2. 观察 df 的数据和结构
#

# %% [markdown]
# ```
#     count     254.000000 #数据数量
#     mean      857.064052 #平均数
#     std       168.881017 #标准差
#     min       572.840027 #最小值
#     25%       709.757492 #较低的百分位数
#     50%       841.630005
#     75%      1008.975006 #较高的百分位数
#     max      1229.910034 #最大值
# ```

# %%
print(df["Adj Close"])


# %%
print(df["Adj Close"].describe())

# %% [markdown]
# 3. 使用 matplotlib 绘制股票市场曲线图
#

# %%

# 加上 上一行的语句可以不使用第12行的语句就生成图片
mpl.rc("figure", figsize=(8, 7))  # 自定义图形的各种默认属性，8行7列
# mpl.__version__
style.use('ggplot')  # ggplot代表图片主题，橙黄色就是图片的主题色
df["Adj Close"].plot(label="TSLA")  # 标签是TSlA
plt.legend()
plt.show()


# %% [markdown]
# 4. 生成移动平均值,确定趋势
#

# %%
df.ta.ema(close="Adj Close", length=10, append=True)  # 数据前length长度计算移动平均值
df


# %%


# %% [markdown]
# ###### Pandas TA 是一个易于使用的库，它建立在 Python 的 Pandas 库的基础上，具有 100 多个指标和实用程序功能。这些指标通常用于列或标签类似于以下内容的金融时间序列数据集：日期时间，开盘价，高价，低价，收盘价，交易量等。其中包括许多常用指标，例如：简单移动平均线（SMA），移动平均线收敛散度（MACD），船体指数移动平均线（HMA），布林线（BBANDS），动平衡量（OBV），Aroon 和 Aroon 振荡器（AROON）等。
#

# %% [markdown]
# 5. 将收盘价与移动平均值一起打印，观察两者之间的关系
#

# %%
# print(df["Adj Close"])  # 便于查看
df["Adj Close"].plot(label="TSLA")  # 用Adj Close那列数据进行画图


# %% [markdown]
# 6. 计算回报偏差
#

# %%
df["Return"] = df["Adj Close"]/df["Adj Close"].shift(1)-1
# shift(1)将数据下降一次
df["Return"].plot(label='return')
plt.show()


# %% [markdown]
# 7. 删除前 10 行数据
#

# %%


# %%
# iloc函数，属于pands库，全称为index location，即对数据进行位置（location）索引（index）。
df = df.iloc[10:]

# %%
df.head(10)  # 输出头10个数据

# %% [markdown]
# 8. 最高价和最低价百分比

# %%
df_reg = df.loc[:, ['Adj Close', 'EMA_10', "Return"]]
# 参数1：利用切片遍历所有标签(从high开始，到Return结束)
# 参数2：以列表的形式对df数据操作，保留列表内的标签数据
df_reg["HL_PCT"] = (df['High']-df["Low"])/df["Close"]*100.0

# %%
df_reg

# %% [markdown]
# 9. 计算收盘变化率

# %%
df_reg["PCT_change"] = (df['Close']-df["Open"])/df["Open"]*100.0


# %%
df_reg

# %% [markdown]
# 10. 预处理
#
# - 在将数据放⼊预测模型之前，我们将对数据进⾏标准化，使每个样本都可以具有相同的线性回归分布。

# %%

# %%
X = np.array(df_reg.drop(["EMA_10"], 1))
# 将ema_10这⼀列去掉，剩余的作为训练数据x


# %%
print(X)

# %%
X = scale(X)
# 对x进⾏标准化 (X-mean)/std


# %%
print(X)

# %%
Y = np.array(df_reg['EMA_10'])


# %%
print(Y)

# %%
X_lately = X[-10:]  # 倒数十条数据
X = X[:-10]  # 除去后十条的数据
Y = Y[:-10]  # 除去后十条的数据


# %%
print(X[:5])
print(X.shape)
print(Y.shape)


# %% [markdown]
# 11. 模型训练

# %%


# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.2, random_state=2022)


# %% [markdown]
# - 进行线性回归

# %%
clfreg = LinearRegression(n_jobs=-1)  # n_jobs表示启动设备cpu的核心个数,-1表示启动所有的核心
clfreg.fit(X_train, Y_train)  # 训练
confidencereg = clfreg.score(X_test, Y_test)  # 利用测试集进行评估


# %%
print('线性回归模型决定系数为 ', confidencereg)


# %%
forecast_set = clfreg.predict(X_lately)  # 进行预测
df_reg["Forecast"] = np.nan  # 建立全是NaN的列表,用来给后面更新


# %%
print(forecast_set.shape)

# %%
df_reg  # 应该是print(df_reg)


# %%
last_date = df_reg.iloc[-1].name  # 索引最后一行数据的行标题


# %%
print(last_date)


# %%
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)  # 数学里面的Δ,变化days=1为一天


# %%
print(next_unix)

# %%
for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    df_reg.loc[next_date] = [np.nan for _ in range(len(df_reg.columns)-1)]+[i]
    print(df_reg.loc[next_date])


# %%
df_reg["Adj Close"].tail(200).plot()
df_reg['Forecast'].tail(200).plot()
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
