# 临近编码 Neighbor Embedding

在非监督学习降维算法中，高纬度的数据，在他附近的数据我们可以看做是低纬度的，例如地球是三维度的，但是地图可以是二维的。
那我们就开始上算法

## 1. Locally Linear Embedding (LLE)
我们需要找到$w_{ij}$ 来最小化：
![在这里插入图片描述](https://img-blog.csdn.net/20181006101300124?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181006101009353?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
找到$w_{ij}$后，我们固定它，然后在z中进行判断
![在这里插入图片描述](https://img-blog.csdn.net/20181006101424812?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
实验：
![在这里插入图片描述](https://img-blog.csdn.net/20181006101541702?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181006101604191?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 2. Laplacian Eigenmaps
我们回一下半监督模型中： 如果x1 和 x2 在高密度空间相似，那么他们的结果y1,y2也形似,S 衡量label 平滑度
![在这里插入图片描述](https://img-blog.csdn.net/2018100610202959?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
如果x1 和 x2 在高密度空间相似，z1 和z2也相似
![在这里插入图片描述](https://img-blog.csdn.net/20181006102411175?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
那么zi和zj等于0 怎么办？我们加入条件限制：
![在这里插入图片描述](https://img-blog.csdn.net/20181006102505424?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 2. T-distributed Stochastic Neighbor Embedding (t-SNE)
前面提到的算法的问题是，相似的数据离得很近，但是很有可能会重叠
![在这里插入图片描述](https://img-blog.csdn.net/20181006102641414?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
(1) t-SNE算法首先计算相似度（x和z分布）
![在这里插入图片描述](https://img-blog.csdn.net/20181006102853709?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181006102917828?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们需要找到z的集合，使得P和Q的相似度KL最小
![在这里插入图片描述](https://img-blog.csdn.net/20181006102929628?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
（2）衡量相似度函数的选择
![在这里插入图片描述](https://img-blog.csdn.net/20181006103145989?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
横坐标是zi和zj的距离
![在这里插入图片描述](https://img-blog.csdn.net/20181006104746461?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
可见t-SNE所使用的相似度函数，在距离增大的过程中，相似度下降较慢，更能区分不同的相似度，但是使用的时候注意，不应该是动态数据，而经常是训练好的静态数据做可视化。
![在这里插入图片描述](https://img-blog.csdn.net/2018100610493438?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
