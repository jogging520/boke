# Support Vector Machine1 (SVM)
支持向量机是机器学习算法中最重要的算法之一。
## 1. 间距（margin） 
![在这里插入图片描述](https://img-blog.csdn.net/20180918143107546?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
在二分类中，我们还是从分类问说起，图中ABC三个点，图中A点比C点更加“自信”，因为A离间距更远。分类正确的可能性更大，我们能不能找到这样的决策线（距离）是的‘ABC’都能够更加自信呢？ SVM就是做这样的工作
2. 函数间距和几何间距 
函数间距：我们定义：当y=1 分类为1，$w^{T}x+b$ 越大，间距越大。那是不是$（w,b）$越大越好呢，不是的！$2*(w,b)$ 虽然让函数间距扩大2北，但是没有任何意义，所以$||w||=1$ 正则化w（normalization）
  ![在这里插入图片描述](https://img-blog.csdn.net/2018091814455298?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
所以，我们的目标是在间距中找到最小的，让最小的间距最大化（看起来有点绕，但是仔细想一下，最小的间距最大化，就是找到了那个最大的分类线）
![在这里插入图片描述](https://img-blog.csdn.net/20180918150804566?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20180918150821141?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
几何间距：
![在这里插入图片描述](https://img-blog.csdn.net/20180918150907521?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
几何间距就更加直观，就是找到一个区分线，让整体效果更好。
就是找到最好的$\gamma$,  其中$w/||w||$是单位向量。 那么有$w^{T}x+b=0$ 上的点就是决策线上的点。所以图中的B点在决策线上，
![在这里插入图片描述](https://img-blog.csdn.net/2018091815213797?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20180918152318769?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
最后还是问题划归为：
![在这里插入图片描述](https://img-blog.csdn.net/20180918152350461?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 2. 优化分类器（The optimal margin classifier）
那我们如何找到这个问题分类器，数学公式如下：
![在这里插入图片描述](https://img-blog.csdn.net/2018091815274389?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

一般的，我们让$\gamma=1$,上面的公式等价为：这就可以用QP软件解二次极值的问题了。
![在这里插入图片描述](https://img-blog.csdn.net/20180918154034575?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 3. 拉格朗日乘子法（Lagrange duality）
拉格朗日乘子法的基本思想就是通过引入拉格朗日乘子来将含有n个变量和k个约束条件的约束优化问题转化为含有（n+k）个变量的无约束优化问题，或者我们可以这么看，拉格朗日乘子法通过将k个约束条件转化进偏导方程组中的k个等式从而使得原问题不再出现约束。**拉格朗日乘子背后的数学意义是其为约束方程梯度线性组合中每个向量的系数。**
![在这里插入图片描述](https://img-blog.csdn.net/20180919085731599?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们用拉格朗日乘法子法表示：
![在这里插入图片描述](https://img-blog.csdn.net/20180919090010575?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
求偏导数
![在这里插入图片描述](https://img-blog.csdn.net/20180919090219271?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
当L取满足上述条件然后取得极值后，我们可以发现有 min(L) = min(f) 也就是说我们只要求出上述方程组，其解便是f的极值点。则我们便将对于f的约束优化问题转化成了对于L的非约束优化问题。

## 3. KKT 条件
KKT条件是指在满足一些有规则的条件下, 一个非线性规划(Nonlinear Programming)问题能有最优化解法的一个必要和充分条件. 这是一个广义化拉格朗日乘子法的成果. 一般地, 一个最优化数学模型的列标准形式参考开头的式子, 所谓 Karush-Kuhn-Tucker 最优化条件。
首先我们看：
![在这里插入图片描述](https://img-blog.csdn.net/20180919091902931?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
广义的拉格朗日乘子法
![在这里插入图片描述](https://img-blog.csdn.net/20180919092041196?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们定义：
![在这里插入图片描述](https://img-blog.csdn.net/20180919092156203?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20180919092329467?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们得到一个等价的公式：
![在这里插入图片描述](https://img-blog.csdn.net/20180919092430179?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20180919092624428?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
所以符合KKT条件：
![在这里插入图片描述](https://img-blog.csdn.net/20180919092926440?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


## 4. 优化间距分类器（Optimal margin classifiers）
我们回归SVM问题：
![在这里插入图片描述](https://img-blog.csdn.net/20180918154034575?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
由于拉格朗日乘子法，我们有：
![在这里插入图片描述](https://img-blog.csdn.net/20180919093321291?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
针对w，b求导数：（w可以计算出来）
![在这里插入图片描述](https://img-blog.csdn.net/20180919093422728?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/2018091909351731?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

然后我们把w带入拉格朗日的方程：
![在这里插入图片描述](https://img-blog.csdn.net/20180919093713380?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20180919093754790?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
所以我们的等价公式为：
![在这里插入图片描述](https://img-blog.csdn.net/20180919094559152?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

经过如此复杂的证明和运算：其实我们得到，w的值是x的内积的线性组合：
![在这里插入图片描述](https://img-blog.csdn.net/20180919095037375?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


