发现了一个Bug  **X[:,k]** 切片产生的数组大小不是(47,1)

numpy 和 pandas 中求 std的方法结果略有不同



**多变量时要注意进行特征放缩**

```python
data = (data - data.mean())/data.std()
```



损失函数
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^i) - y^i)^2
$$

$$
h_\theta(x) = \theta^T x = \theta_0 + \theta_1x_1
$$



梯度下降
$$
\theta_j = \theta_j - \alpha\frac{1}{m}(h_\theta(x^i) - y^i)x_j^i
$$
​	Simulaneously

