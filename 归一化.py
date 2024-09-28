import numpy as np
import pandas as pd

def process_and_save_data(data, cols, file_path, threshold=3):
    # 函数定义：移除异常值
    def remove_outliers(data_series, threshold):
        mean = np.mean(data_series)
        std = np.std(data_series)
        cutoff = std * threshold
        lower_bound = mean - cutoff
        upper_bound = mean + cutoff
        return [x if lower_bound <= x <= upper_bound else None for x in data_series]

    # 函数定义：最小-最大归一化
    def min_max_normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    # 应用异常值移除和前值填充
    for col in cols:
        data[col] = remove_outliers(data[col], threshold)
    data.fillna(method='ffill', inplace=True)

    # 应用最小-最大归一化
    for col in cols:
        data[col] = min_max_normalize(data[col])

    # 保存处理后的数据
    data.to_csv(file_path, index=False)

# 使用此函数的例子
data = pd.read_csv("/home/zly/hjy_code/RGC/dataset/Bai260/Bai260.csv")  # 读取数据
cols = ['AC', 'CNL', 'DEN', 'GR', 'SP', 'RT']      # 定义列名
process_and_save_data(data, cols,"/home/zly/hjy_code/RGC/dataset/Bai260/Bai260.csv")  # 调用函数并保存结果



