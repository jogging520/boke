#朴素贝叶斯
我们先来看贝叶斯公式：
![这里写图片描述](https://img-blog.csdn.net/20180915151050597?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
这个和咱们上一讲生成概率模型的公式是不是很相似，朴素贝叶斯其实就是概率生成模型的一个特例，概率生成模型是假设x 是服从某种特定的概率分布的。x中的各个维度有有相互关系的。 但是朴素贝叶斯为什么朴素，就是假设x是独立分布的。
以邮件分类应用为例，当邮件中出现单词‘buy’，‘price’很可能是广告邮件，我们可能把他分类为垃圾邮件。那么我们得到：
![这里写图片描述](https://img-blog.csdn.net/2018091515193191?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
联合分布中的参数定义:
![这里写图片描述](https://img-blog.csdn.net/20180915152053534?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
所以，我们从新的x中对他的分类的计算：
![这里写图片描述](https://img-blog.csdn.net/20180915152301311?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**这里还有一个技巧：拉普拉斯平滑变换（Laplace smoothing）：**
还是邮件分类的例子，如果我们想给NIP大会投稿，邮件中第一次出现NIP的单词，根据上面的公式：
	 ![这里写图片描述](https://img-blog.csdn.net/20180915153300553?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180915153345773?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这个结果肯定不是我们想要的。
我们把概率的分子加+1，分母加上要分类的数总数k，如果k=2 就是2分类
![这里写图片描述](https://img-blog.csdn.net/20180915153615774?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

所以我们的概率计算为：
![这里写图片描述](https://img-blog.csdn.net/20180915153748997?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

然后使用贝叶斯公式计算就可以。
