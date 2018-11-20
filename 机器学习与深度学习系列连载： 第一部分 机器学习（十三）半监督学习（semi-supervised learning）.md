在实际数据收集的过程中，带标签的数据远远少于未带标签的数据。 我们据需要用带label 和不带label的数据一起进行学习，我们称作半监督学习。

 - Transductive learning：没有标签的数据是测试数据
 - Inductive learning：没有标签的数据不是测试数据
 ![在这里插入图片描述](https://img-blog.csdn.net/20181003063907905?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
为什么没有标签的数据会帮助我们学习呢？ 是因为没有标签数据的分布可能会告诉我们一些潜在的规律。
## 1.半监督生成模型 Semi-supervised Learning for Generative Model

我们回忆一下监督学习的生成模型，计算先验概率，然后通过概率模型估计，计算分类概率。
![在这里插入图片描述](https://img-blog.csdn.net/20181003064303605?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
那半监督的概率生成模型是：

 - 寻找概率最大的$P(C_{i})$ 和$P(x|C_{i})$
 - $P(x|C_{i})$ 符合高斯分布
 ![在这里插入图片描述](https://img-blog.csdn.net/20181003064933406?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 算法流程如下，但是最后的结果影响与初始值的初始化，结构和EM算法类似
 ![在这里插入图片描述](https://img-blog.csdn.net/20181003065359557?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 ![在这里插入图片描述](https://img-blog.csdn.net/2018100306561671?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 2. 低密度分割 Low-density Separation

**大原则：非黑即白**
**（1）Self-training**
![在这里插入图片描述](https://img-blog.csdn.net/20181003140710473?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181003140824655?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**（2）Entropy-based Regularization**
我们估计的分布函数，如何衡量他的好坏
![在这里插入图片描述](https://img-blog.csdn.net/20181003141012986?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
所以我们让他越小越好：
![在这里插入图片描述](https://img-blog.csdn.net/20181003141057315?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
考虑到了Entropy因素，Loss函数最后可以写成
![在这里插入图片描述](https://img-blog.csdn.net/20181003141153486?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**（3）Semi-supervised SVM**
semi-SVM 中，我们假设没有标签的数据可以任意标注
![在这里插入图片描述](https://img-blog.csdn.net/20181003141404236?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们取margin 最大的和error最小的
![在这里插入图片描述](https://img-blog.csdn.net/20181003141543303?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 3. 平滑性假设 Smoothness Assumption
近朱者赤，近墨者黑

假设：相似的x 有着相同的分类

 - x 并不是uniform 统一的
 - 如果$x_{1}$和$x_{2}$在高密度区域中相似，那么他们的结果也就y_{1}$和$y_{2}$一致
 
 ![在这里插入图片描述](https://img-blog.csdn.net/20181003142117532?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
（1）聚类，然后标注 Cluster and then Label
![在这里插入图片描述](https://img-blog.csdn.net/20181003142229135?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
（2）以图为基础的方法 Graph-based Approach
   
![在这里插入图片描述](https://img-blog.csdn.net/20181003142414439?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 很显然，当图中的点能走通，说明是一类。
 创建图的方法（ Graph Construction）：
 
 - 定义$x_{i}$和$x_{j}$的相似度s($x_{i}$,$x_{j}$)
 - 加入边edge
   K Nearest Neighbor
   e-Neighborhood 
   ![在这里插入图片描述](https://img-blog.csdn.net/20181003143001715?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
    - edge 的权重 与s($x_{i}$,$x_{j}$)称比例
    
s($x_{i}$,$x_{j}$)一般表示为Gaussian Radial Basis Function:：
![在这里插入图片描述](https://img-blog.csdn.net/20181003143343290?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
        

 -    定义图的平滑程度 Define the smoothness of the labels 
s 越小越平滑：
![在这里插入图片描述](https://img-blog.csdn.net/20181003143611745?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
如果我们定义s为：
![在这里插入图片描述](https://img-blog.csdn.net/20181003143952462?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181003144303377?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 4. Better Representation
去芜存菁，化繁为简  具体内容我们再降维的章节介绍。（下一节）
