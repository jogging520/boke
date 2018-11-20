# 决策树1（Decision Tree  基础概念）

 - [ ] 每个内部节点对应于某个属性上的测试
 - [ ] 每个分支对应于测试的可能结果
 - [ ] 每个叶节点对应于一个预测的可能结果

主要有：学习过程、预测过程两部分组成

 - 学习过程： 主要是对样本分析来划分属性（内部节点多对应的样本属性）
 - 训练过程： 将测试示例从根节点开始沿着划分属性的“判定测试序列”下行，一直   			   到叶节点。 西瓜书中的图例：
![在这里插入图片描述](https://img-blog.csdn.net/2018092908581014?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##基本流程：

 策略： 分而治之（divided  and  conquer ）
 自根至叶的递归过程
 在每个中间节点寻找一个“划分”（split or test）属性
三种停止条件
 - [ ]  当前节点包含的样本属于同一类别，无需划分
 - [ ]  当前的属性集为空或者所有样本在所有属性的取值相同
 - [ ]  当前节点包含的样本集合为空，不能划分
 算法如下：
 ![在这里插入图片描述](https://img-blog.csdn.net/20180929091204711?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
那我们如何衡量并划分最优属性呢？
## 信息增益 (information gain)

信息熵 (entropy) 是度量样本集合“纯度”最常用的一种指标。假定当前样本集合 D 中第 k 类样本所占的比例为$p_{k}$，则 D 的 信息熵定义为:
![在这里插入图片描述](https://img-blog.csdn.net/2018092909192611?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20180929092104467?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

说了这么多，我们举例说明：
![在这里插入图片描述](https://img-blog.csdn.net/20180929092240880?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/2018092909314584?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
以此类推：
![在这里插入图片描述](https://img-blog.csdn.net/20180929093300927?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 最后我们得到的结果：
 ![在这里插入图片描述](https://img-blog.csdn.net/2018092909345939?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 通过西瓜书的例子，大家对抽象的概念有没有一定认识呢？
 
 ## 增益率 (gain ratio)
 信息增益：对可取值数目较多的属性有所偏好有明显弱点，例如：考虑将“编号”作为一个属性。 我们引入增益率的概念：
 ![在这里插入图片描述](https://img-blog.csdn.net/20180929093751596?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
属性 a 的可能取值数目越多 (即 V 越大)，则 IV(a) 的值通常就越大，算法设计的时候，先从候选划分属性中找出信息增益高于平均水平的，再从中选取增益率最高的

 ## 基尼指数 (gini index)
 ![在这里插入图片描述](https://img-blog.csdn.net/20180929094345280?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 ![在这里插入图片描述](https://img-blog.csdn.net/20180929094557876?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

