# 非监督度学习-1 Unsupervised Learning-1（K-means,HAC,PCA）
非监督学习方法主要分为两大类
 - Dimension Reduction (化繁为简)
 ![在这里插入图片描述](https://img-blog.csdn.net/20181003162048216?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
- Generation （无中生有)
![在这里插入图片描述](https://img-blog.csdn.net/20181003162125997?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
目前我们仅专注化繁为简，降维的方法，无中生有（GAN为代表的）方法，以后关注。
![在这里插入图片描述](https://img-blog.csdn.net/20181003162357577?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 1. Clustering
• K-means 算法
经典的非监督根据距离分类算法：
![在这里插入图片描述](https://img-blog.csdn.net/20181003162657429?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

- Hierarchical Agglomerative Clustering (HAC)
 根据数据两两间的相似度，进行建立一棵树，进行分类
 ![在这里插入图片描述](https://img-blog.csdn.net/20181003162912375?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 2. 分布的重表示 Distributed Representation
![在这里插入图片描述](https://img-blog.csdn.net/20181003163236825?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们主要介绍Principle Component Analysis（PCA）：
需要找到W，$𝑧 = 𝑊𝑥$ 降低维度到 1-D:

**（1）线性代数表示**
![在这里插入图片描述](https://img-blog.csdn.net/20181003163559918?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
使得投影的结果的方差最大化
![在这里插入图片描述](https://img-blog.csdn.net/20181003163648296?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
多维度投影中w1和w2是正交的
![在这里插入图片描述](https://img-blog.csdn.net/20181003163829853?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
数学化证明，PCA与协方差有关 最大化$(w^{1})^{T} cov(x)w^{1}$
![在这里插入图片描述](https://img-blog.csdn.net/20181003164034950?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181003164523254?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181003164602991?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181003164629895?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
因为S是对称矩阵，是半正定，特征值非负。使用拉格朗日乘子法：
![在这里插入图片描述](https://img-blog.csdn.net/20181003164945325?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181003165026269?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
$w^{1}$是特征向量，$\lambda_{1}$是最大的特征值
同理：
![在这里插入图片描述](https://img-blog.csdn.net/20181003165220565?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
$w^{2}$是特征向量，$\lambda_{2}$是第二大的特征值

PCA去相关性举例：
![在这里插入图片描述](https://img-blog.csdn.net/20181003165555510?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181003165730534?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**（2）另一种视角看PCA** 
举例：手写数字是由基本的图片元素组成
![在这里插入图片描述](https://img-blog.csdn.net/20181003165905183?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
那么7是由以下图片元素组成
![在这里插入图片描述](https://img-blog.csdn.net/20181003165952835?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![在这里插入图片描述](https://img-blog.csdn.net/20181003170037880?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181003170215899?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们有：
![在这里插入图片描述](https://img-blog.csdn.net/20181003170522294?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181003170628884?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
PCA可以看看做是特殊的神经网络，元素间是正交的
![在这里插入图片描述](https://img-blog.csdn.net/2018100317093886?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
