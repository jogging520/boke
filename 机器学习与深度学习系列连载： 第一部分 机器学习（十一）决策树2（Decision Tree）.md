# 决策树2
决策树很容易出现过拟合问题，针对过拟合问题，我们采用以下几种方法

## 划分选择 vs 剪枝

剪枝 (pruning) 是决策树对付“过拟合”的 主要手段！

基本策略：

 - 预剪枝 (pre-pruning): 提前终止某些分支的生长
 - 后剪枝 (post-pruning): 生成一棵完全树，再“回头”剪枝
 
剪枝过程中需评估剪枝前后决策树的优劣
我们还是以西瓜书的例子：
![在这里插入图片描述](https://img-blog.csdn.net/20181001152326813?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

我们通过训练集得到未剪枝决策树：
![在这里插入图片描述](https://img-blog.csdn.net/20181001152511683?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
验证数据
![在这里插入图片描述](https://img-blog.csdn.net/20181001152636392?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 1.预剪枝
![在这里插入图片描述](https://img-blog.csdn.net/20181001152754982?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181001152914450?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181001153119142?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 2. 后剪枝
![在这里插入图片描述](https://img-blog.csdn.net/20181001153317623?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181001153417545?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![在这里插入图片描述](https://img-blog.csdn.net/20181001153624596?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181001153703509?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181001153756908?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 3. 两种策略比较

 - 时间开销：

		预剪枝：训练时间开销降低，测试时间开销降低
		后剪枝：训练时间开销增加，测试时间开销降低

 - 过/欠拟合风险：

		预剪枝：过拟合风险降低，欠拟合风险增加
		后剪枝：过拟合风险降低，欠拟合风险基本不变

 - 泛化性能：后剪枝 通常优于 预剪枝

## 4. 连续值处理
基本思路：连续属性离散化
常见做法：二分法 (bi-partition)

 - n 个属性值可形成 n-1 个候选划分
 - 然后即可将它们当做 n-1 个离散属性值处理
![在这里插入图片描述](https://img-blog.csdn.net/20181001154759293?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 5. 缺失值处理
仅使用无缺失的样例？ 对数据的极大浪费

使用带缺失值的样例，需解决：
Q1：如何进行划分属性选择？
Q2：给定划分属性，若样本在该属性上的值缺失，如何进行划分？

		基本思路：样本赋权，权重划分 （半监督学习）

![在这里插入图片描述](https://img-blog.csdn.net/2018100115521724?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181001155648723?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##  6. 从“树”到“规则”

 - 一棵决策树对应于一个“规则集”
 - 每个从根结点到叶结点的分支路径对应于一条规则
![在这里插入图片描述](https://img-blog.csdn.net/20181001160007182?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##  7. 轴平行划分
单变量决策树：在每个非叶结点仅考虑一个划分属性产生“轴平行”分类面
![在这里插入图片描述](https://img-blog.csdn.net/20181001160246343?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

当学习任务所对应的分类边界很复杂时，需要非常多段划分才能获得较好的近似
![在这里插入图片描述](https://img-blog.csdn.net/20181001160435367?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 8. 多变量(multivariate)决策树
多变量(multivariate)决策树
多变量决策树：每个非叶结点不仅考虑一个属性
例如“斜决策树” (oblique decision tree) 不是为每个非叶结点寻找 最优划分属性，而是建立一个线性分类器
![在这里插入图片描述](https://img-blog.csdn.net/2018100116065959?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

更复杂的“混合决策树”甚至可以在结点嵌入神经网络或其他非线性模型
## 9. 决策树常用软件包
ID3, C4.5, C5.0
http://www.rulequest.com/Personal/

J4.8
http://www.cs.waikato.ac.nz/ml/weka/
