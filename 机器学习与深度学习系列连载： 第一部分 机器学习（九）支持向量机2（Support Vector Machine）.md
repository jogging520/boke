# 另一种视角定义SVM：**hinge Loss +kennel trick** 

SVM 可以理解为就是hingle Loss和kernel 的组合
![这里写图片描述](https://img-blog.csdn.net/20180917091514397?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 1. hinge Loss
还是让我们回到二分类的问题，为了方便起见，我们y=1 看做是一类，y=-1 看做是另一类

![这里写图片描述](https://img-blog.csdn.net/20180917094000963?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
他的Loss 函数是分类错误的次数，很显然，这是个离散的值，不可微分，我们需要找到一个等价的Loss
![这里写图片描述](https://img-blog.csdn.net/20180917094226182?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
于是我们出各种等价Loss 函数的图，黑色的是本身的Loss，红色的square Loss，蓝色的的是square Loss+$\sigma$函数的Loss，绿色的是corssEntropy+$\sigma$函数的Loss（注意横坐标是$y^{n}*f(x)$ 数值越大，Loss越小，说明分类越正确，可以结合前一篇的几何意义理解）

![这里写图片描述](https://img-blog.csdn.net/20180917094338908?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

在这里，我们引入hinge Loss，它的公式：
![这里写图片描述](https://img-blog.csdn.net/20180917095036465?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

含义是：当分类是1，$y=1$，需要最大化0与$1-f(x)$的值,$f(x)>1$,$f(x)$ 比1越大越好；
			  当分类是1，$y=1$，需要最大化0与$1+f(x)$的值,$f(x)<-1$,$f(x)$ 比-1越x小越好
![这里写图片描述](https://img-blog.csdn.net/20180917095054728?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

所以它的Loss图像为（紫色的线段）：

![这里写图片描述](https://img-blog.csdn.net/2018091709551011?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
所以 线性的SVM 的解法完全可以用 gradient decent 解：
![在这里插入图片描述](https://img-blog.csdn.net/20180920084727180?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们的Loss 函数：
![在这里插入图片描述](https://img-blog.csdn.net/20180920085734982?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
然后，这些公式是不是似曾相识：
![在这里插入图片描述](https://img-blog.csdn.net/20180920085841591?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 2. Kernel
通常，我们需要对我们的数据在高维空间进行相关映射。
![在这里插入图片描述](https://img-blog.csdn.net/20180920092342766?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


我们通过拉格朗日乘子法，得到最优的w是 x的线性组合
![在这里插入图片描述](https://img-blog.csdn.net/20180920090139743?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
因为$a_{n}^{*}$很有可能非常稀疏， 非0的$a_{n}^{*}$ 就是支持向量。
![在这里插入图片描述](https://img-blog.csdn.net/20180920090608925?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
第一步：我们做等价转换
![在这里插入图片描述](https://img-blog.csdn.net/20180920090740959?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20180920090755464?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
第二步： 最小化Loss
![在这里插入图片描述](https://img-blog.csdn.net/20180920091008429?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们就是要找到最适合的$a_{n}^{*}$
![在这里插入图片描述](https://img-blog.csdn.net/20180920091056645?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**我们甚至可以不需要知道x的向量，就可以计算与z的内积 $k(x,z)$, 它就是kernel**
![在这里插入图片描述](https://img-blog.csdn.net/20180920091644901?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20180920091702877?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
有很多kernel的例子：Sigmoid Kernel
![在这里插入图片描述](https://img-blog.csdn.net/20180920091958822?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20180920092047175?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
画成图： 是不是就是一个神经网络呢？  恍然所思，殊途同归！
![在这里插入图片描述](https://img-blog.csdn.net/20180920092112289?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

