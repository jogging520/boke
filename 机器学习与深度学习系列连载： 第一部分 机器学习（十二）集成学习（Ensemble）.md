# 集成学习（Ensemble）
## 1. Bagging
我们考虑当结果的 variance 很大，如果降低 variance。
我们可以考虑“平行宇宙”，不同的training set 中生成不同的模型，然后做平均或者voting。
![在这里插入图片描述](https://img-blog.csdn.net/20181002213200991?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)![在这里插入图片描述](https://img-blog.csdn.net/20181002213217797?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 2. Decision Tree（Review）
我们复习上一节的决策树的概念。
![在这里插入图片描述](https://img-blog.csdn.net/20181002213704888?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
来一个有意思的实验，分辨出漫画人物
![在这里插入图片描述](https://img-blog.csdn.net/20181002213754418?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
当单棵决策树的深度为20的时候，得到的结果已经很不错了，但是很有可能会出现一个结果：overfitting。  如果解决overfitting呢？我们看随机森林

## 3. 随机森林（Random Forest）
Decision tree 很容易在训练数据中误差为0，但是产生overfitting。
Random Forest 就是 bagging of decision tree ，是众多决策树的集合。![在这里插入图片描述](https://img-blog.csdn.net/201810022144047?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们使用没有选择的数据做validation数据
![在这里插入图片描述](https://img-blog.csdn.net/20181002214457500?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 4. Boosting
对于Boosting 我们有：
 - 当我们使用机器学习算法得出的分类器的错误率在训练数据中小于50%
 - 我们使用Boosting 可以让最后的错误结果达到0%
Boosting 的框架结构：
	- 首先获得分类器$f_{1}(x)$
	- 找到另一个分类器$f_{2}(x)$ 来帮助$f_{1}(x)$
			- 但是， 如果$f_{2}(x)$与$f_{1}(x)$相似，对于结果的帮助不太大
			- 如果我们想让$f_{2}(x)$成为$f_{1}(x)$的补充（我们将怎样去做）
   - 找到第二个分类器$f_{2}(x)$ 
   - ... 最后集成所有的分类器
   - 所有的分类学习都是序列的

（**1）怎样获取不同的分类器？**

 - 在不同的训练数据集中进行训练
 - 获得不同训练数据集的方法
 		- 重新抽样数据集
 		- 给数据集的数据分配权重
 		- 在实作中，仅仅需要修改cost 函数
 	![在这里插入图片描述](https://img-blog.csdn.net/20181002215903494?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

**（2） Adaboost的思路（Idea of Adaboost）**
思路： 分类器$f_{1}(x)$的错误分类小于50%，我们调整训练数据权重，是的$f_{2}(x)$ 中的训练数据权重，在$f_{1}(x)$出错的地方提高，$f_{1}(x)$正确的地方降低。


![在这里插入图片描述](https://img-blog.csdn.net/2018100222071650?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
究竟训练数据的权重增大或者降低多少呢？？？
![在这里插入图片描述](https://img-blog.csdn.net/20181002220803926?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
推导过程：
![在这里插入图片描述](https://img-blog.csdn.net/20181002221137328?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**（2） Adaboost算法**
![在这里插入图片描述](https://img-blog.csdn.net/20181002221402819?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181002221614128?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
举例说明：
![在这里插入图片描述](https://img-blog.csdn.net/20181002222119556?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181002222143526?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 5.General Formulation of Boosting
![在这里插入图片描述](https://img-blog.csdn.net/20181002222934549?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 6.Stacking
Voting
![在这里插入图片描述](https://img-blog.csdn.net/20181002223313681?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181002223327309?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
