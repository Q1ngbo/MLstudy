

​	一开始使用的Numpy 中的array写的，后来看了黄博提供的代码之后改用Matrix, 已经弃用了···

> Matrix是Array的一个小的分支，包含于Array。所以matrix 拥有array的所有特性。Matrix必须是二维的

Matrix的 **优势**主要有： **相对简单的乘法运算符号**  a 和 b如果是两个矩阵  那a * b 可以替代 a.dot(b)



```python
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
```

可以将DataFrame中数据转为矩阵

theta 在一行中

 