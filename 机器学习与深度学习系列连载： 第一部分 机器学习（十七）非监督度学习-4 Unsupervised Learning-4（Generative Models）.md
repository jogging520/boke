# 生成模型 Generative Models
用非监督学习生成结构化数据，是非监督模型的一个重要分支，本节重点介绍三个算法： Pixel RNN  ，VAE 和GAN（以后会重点讲解原理）

## 1. Pixel RNN
RNN目前还没有介绍，，以后会重点讲解，大家目前认为他是一个神经网络即可
![在这里插入图片描述](https://img-blog.csdn.net/20181006110428799?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
举例：用Pixel RNN 生成怪物精灵；
我们首先进行配色编码：
![在这里插入图片描述](https://img-blog.csdn.net/20181006110847822?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
然后遮盖部门图片，进行图片生成
![在这里插入图片描述](https://img-blog.csdn.net/20181006110747981?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 2.   Variational AutoEncoder(VAE) 

(1) 首先我们先看看AutoEncoder的缺点	：
如图，AutoEncoder能够生成满月和玄月之间的月亮的图片吗，答案是不行。因为两个图片转换为code的中间地带，我们无法控制。
![在这里插入图片描述](https://img-blog.csdn.net/20181006111056460?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
（2）架构过渡：
![在这里插入图片描述](https://img-blog.csdn.net/20181006111337416?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
看到如此复杂的结构是不是很晕，我们一一进行剖析：我们注意到，我们不仅要最小化生成数据和原始数据之间的差距，我们还要最小化上图中黄框中的公式，这是为什么？
（1）从直觉上理解：我们需要在训练数据中加一些noisy ，但是这些噪声数据的方差是我们未知的，使用exp函数将方差非负，然后减去（1+方差），得到如图绿曲线，方差变小。m的平方可以理解为L2函数。
![在这里插入图片描述](https://img-blog.csdn.net/20181006112247954?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
（2）原理
回归的问题本质：我们其实就是想估计概率的分布
	![在这里插入图片描述](https://img-blog.csdn.net/20181006112519458?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们引入混合高斯模型的概念：
![在这里插入图片描述](https://img-blog.csdn.net/2018100611264454?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181006112653958?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
在VAE中，我们假设z符合高斯分布，但是z具体高斯分布的参数未知，我们需要通过神经网络进行估计，每一个z代表的x的一个分布。
![在这里插入图片描述](https://img-blog.csdn.net/20181006113036658?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
整体图：
![在这里插入图片描述](https://img-blog.csdn.net/20181006113122128?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们求概率最大化：
![在这里插入图片描述](https://img-blog.csdn.net/2018100611365812?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
我们需要另外的一个概率分布q(z|x)
![在这里插入图片描述](https://img-blog.csdn.net/20181006113831915?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
然后有：
![在这里插入图片描述](https://img-blog.csdn.net/20181006114000707?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
数学证明：
![在这里插入图片描述](https://img-blog.csdn.net/20181006114245442?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/2018100611425863?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/2018100611431045?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181006114332269?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
VAE的问题：
![在这里插入图片描述](https://img-blog.csdn.net/20181006114606214?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
VAE生成的结果一般都是训练集的结果或者训练集结果的叠加，但是不能生成新的结果，我们可以总结为“一直在模型，从未被超越”。接下来我们简单介绍下GAN（我们以后会详细介绍）

## 3.GAN
拟态演化，到GAN
![在这里插入图片描述](https://img-blog.csdn.net/20181006114724608?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
