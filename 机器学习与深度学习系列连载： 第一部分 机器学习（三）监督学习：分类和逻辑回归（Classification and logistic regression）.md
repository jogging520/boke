#分类和逻辑回归（Classification and logistic
#regression）
我们接着线性回归的问题，在实际问题中，我们不仅需要得出具体的预测数值，我们还需要将数据进行分类。例如，垃圾邮件识别程序，需要将邮件识别为正常邮件（标记为+1），垃圾邮件（标记为 0）。这是一个典型的分类问题。

##逻辑回归（ logistic）
我们拿垃圾邮件二分类（c1（正常）,c2（垃圾））举例，需要找到一个概率$p(c1|x)$，当$p(c1|x)>0.5$ 时候是分类c1，当$p(c1|x)<0.5$的时候的分类是c2。
这个时候我们就找到一个回归函数。
###回归函数
**$z=w*x+b$**   当 $\sigma (z)$ 输出大于0.5时候为C1，小于0.5的时候为c2 
![这里写图片描述](https://img-blog.csdn.net/2018091109212122?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们又开始机器学习三板斧了：

 - 第一步：定义一个函数集合

![这里写图片描述](https://img-blog.csdn.net/20180911092508650?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 - 第二步：判断一个函数的好坏

我们的数据集是（x,C） （输入x，x的分类C（正常邮件，垃圾邮件））
![这里写图片描述](https://img-blog.csdn.net/20180911092753949?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
接下来我们定义逻辑回归好坏的判定公式：给一组我w，b，我们针对每一组数据的概率，他们的乘积就是同时发生的概率（概率论）
![这里写图片描述](https://img-blog.csdn.net/20180911092856930?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们找到的w,b 就是
![这里写图片描述](https://img-blog.csdn.net/20180911093616603?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
通过推导，公式取对数，前面再加符号，等价于把乘法变成加法，求最大值，也变成了求最小值。(crossentropy的概念也就推导出来）
![这里写图片描述](https://img-blog.csdn.net/20180911093758551?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
熵的概念本来是热力学的一个概念，描述物质的混乱程度。 在这里，我们用交叉熵的概念来描述两组不同概率数据分布的相似程度，越小越相似。（这个概念在机器学习中非常重要）
![这里写图片描述](https://img-blog.csdn.net/20180911094058660?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 - 第三步：找到最好的函数（Gradient Decent）
 - ![这里写图片描述](https://img-blog.csdn.net/2018091109491926?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 更新w：

在这里我们惊奇的发现，逻辑回归和线性回归的第三步骤的推导结果是一致的。
我们把线性回归和逻辑回归做一个小小的总结：

 1. 他们的函数选择集不一样
 2. 衡量函数好坏的算法不一样（crossentorpy 和 LMS）
 3. 但是梯度下降的微分函数的结构是一致的

![这里写图片描述](https://img-blog.csdn.net/20180911095413477?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
那我们能不能用线性回归的step2，最小化二次函数来找到逻辑回归最小值
![这里写图片描述](https://img-blog.csdn.net/20180911133121935?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
从以上公式可以看出，在结果附近的微分值都是0，我们作图得到
![这里写图片描述](https://img-blog.csdn.net/20180911133232736?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
所以术业有专攻，分类的问题，还是crossentropy 专业！

##Generative（几率模型）和Discrimitive（逻辑回归）
（留在几率模型中讲，G做了某些假设）




##多分类（Multi-class Classification）
多分类和是二分类的进阶，与二分类一样我们有 z 函数，
![这里写图片描述](https://img-blog.csdn.net/20180911130542299?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
经过softmax层后，分类结果为，我们的目标还是要缩小crosseentropy的大小
![这里写图片描述](https://img-blog.csdn.net/20180911130740846?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

那究竟什么是softmax，就是输入x，经过线性变换到z，然后求e的z次方的值，最后算归一化结果（统一除以$e^{z}$的和）。

在cs229中的证明如下：
![这里写图片描述](https://img-blog.csdn.net/20180911132236113?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180911132248655?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180911132308299?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180911132326201?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/2018091113234654?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180911132355480?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
