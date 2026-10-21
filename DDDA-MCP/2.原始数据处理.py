import pandas as pd
import numpy as np
import os

# 当前脚本所在文件夹
base_dir = os.path.dirname(os.path.abspath(__file__))

# 读取原始Excel文件
df = pd.read_excel(os.path.join(base_dir, '汇总数据-720h.xlsx'))

# 提取所需列：前6列是x1-x6，第7列是y（pandas索引从0开始）
x1 = df.iloc[:, 0]
x2 = df.iloc[:, 1]
x3 = df.iloc[:, 2]
x4 = df.iloc[:, 3]
x5 = df.iloc[:, 4]
x6 = df.iloc[:, 5]
y = df.iloc[:, 6]

# 计算对数空间数据
X1 = np.log10(1 - x1)
X2 = np.log10(x2 * 2 / 1e-6)  # 即 log10(x2 * 1e6)
X3 = np.log10(x3 * 1000 / 101)
X4 = np.log10(x4 / 101)
X5 = np.log10(x5)
X6 = np.log10(x6)
Y = np.log10(y - x1 + 1)

# 构建结果DataFrame
result_df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4,
    'X5': X5,
    'X6': X6,
    'Y': Y
})

# 导出到Excel文件
result_df.to_excel(os.path.join(base_dir, '对数空间数据-720h.xlsx'), index=False)
print('处理完成，结果已保存到 对数空间数据-720h.xlsx')
