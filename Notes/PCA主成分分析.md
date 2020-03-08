# PCA主成分分析

全称 Principal Components Analysis



## PCA有什么用

* 数据可视化 Visualization
* 数据压缩, 加速算法
* 减少特征数量
* 防止过拟合（最好别用）



## PCA怎么用

* 首先，进行均值归一化，计算所有特征均值并减去，如果特征不在一个数量级上，除以标准差

* 计算协方差矩阵
  $$
  \Sigma = \frac{1}{m} \sum_{i=1}^{n}(x^{(i)})(x^{(i)})^T = \frac{1}{m}X^TX
  $$
  

* 计算协方差矩阵的特征向量

  * 使用奇异值分解svd

* 由特征向量计算新特征

  如果想由n维降至k维，需要选取U的前k个向量, U 是n x k维，x 是 n x 1 维
  $$
  z^{(i)} = U_{reduce}^T*x^{(i)}
  $$



## 主成分数量的选取

遵循下面这个
$$
\frac{\frac{1}{m}\sum_{i=1}^m||x^{(i)} - x_{approx}^{(i)}||^2}{\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}||^2} = 1 - \frac{\sum_{i=1}^{k}s_{ii}}{\sum_{i=1}^{n}s_{ii}}<=0.01 (或者0.05)
$$


比例小于0.01 代表着原本数据0.99的特征都保留了, 第二个使用svd 所得S(对角矩阵)计算



## 怎么还原到原始数据

$$
x_{approx} = U_{reduce}·z
$$

Approximately