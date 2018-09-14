##1.误差分析（Bias and Variance）

当我们以非常复杂的模型去进行测试的时候，可能得到的结果并不理想
![这里写图片描述](https://img-blog.csdn.net/20180912091035408?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
影响结果的主要有两个因素：**Bias 偏差、Variance 方差**

 - Bias 偏差

 在这里，我们定义偏差是指与目标结果的偏移量，这个偏移量是我们选出来的函数的期望 $E(f^{*})$。如图所示：与目标距离远的是大偏差，与目标距离近的是小偏差
 	 ![这里写图片描述](https://img-blog.csdn.net/20180912095310477?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 
 - Variance 方差
而方差描述的的是我们选出来的函数，他的稳定性，是否集中在目标区域
与相对分散的是高方差，相对集中的是低方差
![这里写图片描述](https://img-blog.csdn.net/20180912095656767?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

小总结：偏差描述的是与目标的距离，而方差描述的是分散程度，我们的目的是在**机器学习三板斧**过后，找到一个低偏差，低方差的函数。如图左一
![这里写图片描述](https://img-blog.csdn.net/20180912095839160?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

但是说起来容易，做起来难，我们一起看看吴恩达博士在cs229中的建议

##2. 模型调优
问题举例：以朴素贝叶斯（我们后面会具体讲这个模型）为模型的邮件分类系统，错误率达到了20%，这是不能接受
![这里写图片描述](https://img-blog.csdn.net/20180912102406535?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
吴恩达的建议是：
**1.获得跟多的的训练数据**
**2.尝试更少的更多的特征维度**
**3.尝试更多以的更少的特征维度**
**4.尝试更换邮件头或者邮件体的特征**
**5.把梯度下降的方法运行更多次**
**6.尝试使用牛顿方法**
**7.使用不同的参数$\lambda $**
**8 抛弃朴素贝叶斯模型尝试SVM**

![这里写图片描述](https://img-blog.csdn.net/20180912103208120?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这么多方法，我们怎么选？
误差分析这个工具就可以派上用场了。一般情况下：高方差表示的就是训练误差小于测试误差，高误差表示 训练误差本来就很高

![这里写图片描述](https://img-blog.csdn.net/20180912104342180?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
在这里我们给出典型的学习曲线：

 1. 高方差： 训练误差明显大于训练误差

![这里写图片描述](https://img-blog.csdn.net/20180912104750779?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 1. 高误差： 训练误差大于我们期望的表现，但是与测试误差接近

![这里写图片描述](https://img-blog.csdn.net/20180912104945860?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

所以刚才我们提到的8个方法中的前4个要解决的问题是：
**1.获得跟多的的训练数据** （高方差）
**2.尝试更少的更多的特征维度**（高方差）
**3.尝试更多以的更少的特征维度**（高误差）
**4.尝试更换邮件头或者邮件体的特征**（高误差）

接下来我们假设：
(1)  贝叶斯回归正常邮件的错误率是2%，垃圾邮件的错误率是2%。（如果把2%的正确邮件当成垃圾的，就是不可接受的）
(2) SVM 方法在垃圾邮件的识别错误率是10%，而正常邮件的识别错误率是0.01%。（这个表现是可以接受的）
(3) 但是，你又想用逻辑回归，应为计算量相对比较小

这个问题怎么破？
![这里写图片描述](https://img-blog.csdn.net/2018091210540688?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们拿贝叶斯和SVM两个模型做对比 
![这里写图片描述](https://img-blog.csdn.net/20180912110628204?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们比较最后结果的的正确性
![这里写图片描述](https://img-blog.csdn.net/20180912110752655?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
当![这里写图片描述](https://img-blog.csdn.net/20180912110828475?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
举例来说：上面的方程表示svm的表现好于beyesian
接下来我们就分析beyesian的损失函数  这里的=$ J(\theta)$想到于负的Loss，$ J(\theta)$越大，相当于Loss越小。
![这里写图片描述](https://img-blog.csdn.net/20180912111128184?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
然后我们比较svm和beyesian的损失函数

 - case1：SVM的正确性大于BLR ，评价函数的结果也大于BLR（Loss小）
  ![这里写图片描述](https://img-blog.csdn.net/20180912111353886?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
  **这里说明BLR模型的整体效果不如SVM，原因是$ J(\theta)$没有找到最大值，还有优化空间，需要用不同的方法优化$ J(\theta)$**
  
 - case2: SVM的正确性大于BLR，但是评价函数的结果小于BLR

![这里写图片描述](https://img-blog.csdn.net/20180912112259782?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
这里说明BLR已经找到了评价函数$ J(\theta)$的最大值，但是结果不理想，需要调整评价函数或者换不同的模型。

所以刚才我们提到的8个方法中的后4个要解决的问题是：

**5.把梯度下降的方法运行更多次**（优化求极值的方法）
**6.尝试使用牛顿方法**（优化求极值的方法）
**7.使用不同的参数$\lambda $**（改变评价函数）
**8 抛弃朴素贝叶斯模型尝试SVM**（改变评价函数）

着这里我们把吴恩达博士给的八个方法具体用到哪些场景已经了“心中有数”了。

