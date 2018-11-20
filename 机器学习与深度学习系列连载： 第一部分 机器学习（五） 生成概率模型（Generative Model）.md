#生成概率模型（Generative Model）

##1.概率分布
我们还是从分类问题说起：
当我们把问题问题看做是一个回归问题， 分类是class 1 的时候结果是1
 分类为class 2的时候结果是-1；
 测试的时候，结果接近1的是class1 ，结果接近-1的是class2
![这里写图片描述](https://img-blog.csdn.net/20180914090430793?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
问题解决了！ 但是这只是看起来很美，但是如果结果远远大于1的时候，他的分类应该是class1还是class2，我们为了降低整体误差，需要调整已经找到的分类函数，这样会实际导致结果的不准确
![这里写图片描述](https://img-blog.csdn.net/20180914090553390?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
所以这是另一个角度，分类不能用回归的思路去做的原因。
##分类问题机器学习三板斧

 - 1.函数（Model）

以二分类为例，$f(x)$
![这里写图片描述](https://img-blog.csdn.net/20180914091131922?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 - 2.损失函数（Loss）

![这里写图片描述](https://img-blog.csdn.net/20180914091253244?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 - 3.找到最好的函数（SVM，perceptorn）（以后会讲）

**我们开始我们的生成概率模型**，首先举一个例子，有两个盒子，有蓝球和绿球，
那么问题来了，如果闭着眼睛拿出一个蓝色的球，并且它盒子1的概率是多少。
$p(B_{1}|x)$, 当$p(B_{1}|x)》0.5$ 我们就认为它属于盒子1.
![这里写图片描述](https://img-blog.csdn.net/20180914091806677?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
在这里盒子1的和盒子2 的先验概率已知。其他概率很好计算。
那么蓝球来自盒子1的概率是：
![这里写图片描述](https://img-blog.csdn.net/20180914092355283?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
推而广之：如果我们看成分类，两个类别
![这里写图片描述](https://img-blog.csdn.net/20180914092601418?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
那么给一个x，他的分类的概率是：
![这里写图片描述](https://img-blog.csdn.net/20180914092654663?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
整体$p(x)$的概率是：
![这里写图片描述](https://img-blog.csdn.net/20180914092816348?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

生成概率模型其实是先假设数据的概率分布（正太、伯努利、泊松），然后用概率公式去计算x所属于的类型$p(C_{1}|x)$

一般的，我们假设x的分布为高斯分布（最为常见的概率分布模型），为什么会往往选择高斯分布呢，概率论中的中心极限定理告诉我们答案。
一维的概率分布一般是钟形曲线，大家都比较了解，那么高纬的分布是：
![这里写图片描述](https://img-blog.csdn.net/20180915091215422?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
均值为$\mu$,协方差为$\sum$
![这里写图片描述](https://img-blog.csdn.net/20180915091605924?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
上面三幅图表示均值都为0，但是协方差分别为为I ， 0.6I，2I

更多的例子
![这里写图片描述](https://img-blog.csdn.net/20180915091729102?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![这里写图片描述](https://img-blog.csdn.net/20180915091757989?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们假设数据点服从高纬高斯分布，那么，我们需要找到这个高斯分布的函数，也就是为$\mu$,和协方差$\sum$。
这个函数满足，它的所有数据点的生成概率是最大的，假设有79个数据点，他的高斯函数的求法：
![这里写图片描述](https://img-blog.csdn.net/20180915092124980?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##2.解决问题
让我们开始我们的分类问题：
我们要进行二分类，分别是水系的怪物精灵和一般的怪物精灵，我们计算得到他们的高斯分布分别为
![这里写图片描述](https://img-blog.csdn.net/20180915092405102?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
那么我们就可以用本篇第一部分的公式计算x的分类了，$p(C_{1})$，$p(C_{2})$ 分别在数据中就可以简单计算，$p(x|C_{1})$，$p(x|C_{2})$ 由它们概率密度函数推导求解得到（积分）
![这里写图片描述](https://img-blog.csdn.net/20180915092814123?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
这样就把分类问题变成了一个概率计算问题了。
但是结果不理想，只有54%的正确率。 	
![这里写图片描述](https://img-blog.csdn.net/20180915093329164?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)。
我们分析一下原因，是由于两类额协方差导致参数过多，那我们让协方差共享$\sum$,减少协方差的种类。
![这里写图片描述](https://img-blog.csdn.net/20180915093552871?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们得到了73%的正确率
![这里写图片描述](https://img-blog.csdn.net/20180915093635904?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
