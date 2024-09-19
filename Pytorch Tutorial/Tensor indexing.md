
```python
import torch
import numpy as np

# Tensor indexing 是指在张量（tensor）中通过索引来访问或操作其特定部分的过程。
# 张量是一种多维数组，用于表示多维数据，例如向量、矩阵或更高维的数组。

# 在使用张量索引时，通常会用到以下几种基本操作：

# 1. 基本索引（Basic Indexing）:
# 使用整数索引来访问特定位置的元素。例如，在二维张量（矩阵）中，可以使用 tensor[i, j] 来访问第 i 行、第 j 列的元素。

# 2. 切片（Slicing）:
# 使用冒号（:）表示切片，可以获取张量的子部分。例如，tensor[:, 0] 表示获取所有行的第一列。

# 3. 高级索引（Advanced Indexing）:
# 使用整数数组或布尔数组来进行索引。例如，tensor[[0, 1, 2], [0, 1, 2]] 表示获取张量中 (0,0), (1,1), (2,2) 位置的元素。
# 布尔索引可以用于筛选元素，例如 tensor[tensor > 0] 获取张量中所有大于0的元素。

# 4. 使用 ...（Ellipsis）:
# 用于表示省略的维度。例如，tensor[..., 0] 表示获取最后一个维度的第一个元素，其余维度保持不变。

# 5. 索引赋值（Index Assignment）:
# 可以使用索引直接修改张量的部分元素。例如，tensor[0, :] = 1 表示将张量第一行的所有元素设置为1。

# 这些操作在深度学习框架（如 PyTorch、TensorFlow）中非常常见，方便进行张量操作和数据处理。

batch_size = 10  # 批量大小，即每次训练所使用的样本数量
features = 25  # 特征数量，即每个样本的特征数

# 创建一个形状为 (batch_size, features) 的张量 x，其中包含随机生成的数值
# 创建了一个形状为 (10, 25) 的张量 x，其中包含从标准正态分布中随机生成的数值。
# batch_size 是 10，表示有10个样本；features 是 25，表示每个样本有25个特征。
x = torch.randn((batch_size, features))

# 打印张量 x 的第一个样本（第 0 行）
print(x[0])
print("x[0,:] shape is: " +  str(x[0].shape))
print("x[:,0] shape is: " + str(x[:,0].shape))
print(x[2, 0:10])# 打印张量 x 的第三个样本的前10个特征

# Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand(3, 5)
rows = torch.tensor([1, 0]) # 取第2"行"和第1"行"的元素
cols = torch.tensor([4, 0]) # 取第5"列"和第1"列'的元素
print(x[rows, cols].shape)

# More Advanced Indexing
x = torch.arange(11)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# Useful Operations
# Condition Operations Indexing
#  torch.where 函数：
# torch.where(condition, x, y) 函数的作用是：
# 对于 condition 中为 True 的位置，选择 x 中对应位置的元素。
# 对于 condition 中为 False 的位置，选择 y 中对应位置的元素。
# 对于张量 x 中大于 5 的元素，保留原值；对于不大于 5 的元素，将它们的值乘以 2，并打印结果。
print(torch.where(x > 5, x, x*2))

# 创建一个包含重复元素的张量，使用unique函数去重，并打印去重后的结果。
# 最终打印的结果是 tensor([0, 1, 2, 3, 4, 5])，即原始张量中的唯一元素。
print((torch.tensor([0, 0, 1, 1, 2, 3, 4, 4, 5])).unique())

# 用于打印张量 x 的维度数量。具体来说，它告诉你这个张量是几维的（例如，1D，2D，3D 等）。
# 这是一个很有用的函数，特别是在处理高维数据时，可以帮助你了解张量的结构。
# 1D 张量
# x1 = torch.tensor([1, 2, 3])
# print(x1.ndimension())  # 输出: 1
# 2D 张量
# x2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(x2.ndimension())  # 输出: 2
# 3D 张量
# x3 = torch.randn(3, 4, 5)
# print(x3.ndimension())  # 输出: 3
print(x.ndimension())

print(x.numel())
```


