# *欢迎进入机器学习的世界* 

本教程是根据台湾大学李弘毅老师的课程[机器学习](http://speech.ee.ntu.edu.tw/~tlkagk/%20%E2%80%9C%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E2%80%9D)课程，斯坦福大学[CS229](http://cs229.stanford.edu/)、[CS231N](http://cs231n.stanford.edu/)、[CS224N](https://web.stanford.edu/class/cs224n/)、[CS20i](http://web.stanford.edu/class/cs20si/syllabus.html)、伦敦大学学院 ([UCL-Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)课程，翻译、总结、提炼，将零星的知识点、算法进行串接，并加入个人的理解，形成机器学习基础理论、图像处理、自然语言处理、强化学习、对抗学习的整体知识框架的入门、提高教程。

在本教程最开始的地方，首先忠心感谢这些高水平课程，本人是经过反复观看（至少十次）、思考、编码，才获得较浅层次领悟（本教程中也会引用这些课程的经典内容、图片、代码，引用的时候我也会具体注明）。
![这里写图片描述](https://img-blog.csdn.net/20180902230014846?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


## *1.编写目的：*

 - **突破语言障碍**：机器学习、深度学习核心课程、算法、论文都是英文。机器学习爱好者可能在语言上望而却步，而内容全面、高水平的中文教程相对较少。
 - **内容全面**：各类机器学习中文学习笔记比较多，但是只是针对某个算法或者某门课程（方向），整体上将机器学习理论、图像、自然语言处理、强化学习、对抗网络算法和最新成果进行串联的中文教程较少。
 -  **通俗易懂**：用“最通俗的语言、最少的数学公式”，带领徘徊在机器学习门口的同学们，入门、提升、掌握机器学习基础理论、掌握深度学习的核心理念、算法`。
##*2.读者要求：*
建议学习本教程的同学具备一定的高等数学、概率论、线性代数的知识和掌握Python语言。

## *2.学习路径：*
本教程一共分为五大部分，估计在50篇博文左右（每周一更或者两更）：
-   **第一部分：机器学习（已经完成）**
01.[机器学习与深度学习系列连载（NTU-Machine Learning, cs229, cs231n, cs224n, UCL-RL，cs20i:欢迎进入机器学习的世界](https://blog.csdn.net/dukuku5038/article/details/82253966)  
02.[机器学习与深度学习系列连载：第一部分机器学习（一）导论](https://blog.csdn.net/dukuku5038/article/details/82347021)  
03.[机器学习与深度学习系列连载：第一部分机器学习（二）监督学习：线性回归](https://blog.csdn.net/dukuku5038/article/details/82503111)  
04.[机器学习与深度学习系列连载：第一部分机器学习（三）监督学习：分类和逻辑回归 Classification and logistic regression](https://blog.csdn.net/dukuku5038/article/details/82585523)  
05.[机器学习与深度学习系列连载：第一部分机器学习（四）误差分析（BiasandVariance）和模型调优](https://blog.csdn.net/dukuku5038/article/details/82682855)  
06.[机器学习与深度学习系列连载：第一部分机器学习（五）生成概率模型（GenerativeModel）](https://blog.csdn.net/dukuku5038/article/details/82698867)  
07.[机器学习与深度学习系列连载：第一部分机器学习（六）训练数据和测试数据（TraindataandTestdata）](https://blog.csdn.net/dukuku5038/article/details/82699481)  
08.[机器学习与深度学习系列连载：第一部分机器学习（七）朴素贝叶斯（NaiveBayes）](https://blog.csdn.net/dukuku5038/article/details/82714617)  
09.[机器学习与深度学习系列连载：第一部分机器学习（八）支持向量机1（SupportVectorMachine）](https://blog.csdn.net/dukuku5038/article/details/82746437)  
10.[机器学习与深度学习系列连载：第一部分机器学习（九）支持向量机2（SupportVectorMachine）](https://blog.csdn.net/dukuku5038/article/details/82767724)  
11.[机器学习与深度学习系列连载：第一部分机器学习（十）决策树1（DecisionTree）](https://blog.csdn.net/dukuku5038/article/details/82781286)  
12.[机器学习与深度学习系列连载：第一部分机器学习（十一）决策树2（DecisionTree）](https://blog.csdn.net/dukuku5038/article/details/82917083)  
13.[机器学习与深度学习系列连载：第一部分机器学习（十二）集成学习（Ensemble）](https://blog.csdn.net/dukuku5038/article/details/82929068)  
14.[机器学习与深度学习系列连载：第一部分机器学习（十三）半监督学习（semi-supervisedlearning）](https://blog.csdn.net/dukuku5038/article/details/82932618)  
15.[机器学习与深度学习系列连载：第一部分机器学习（十四）非监督度学习-1 UnsupervisedLearning-1](https://blog.csdn.net/dukuku5038/article/details/82932618)  
16.[机器学习与深度学习系列连载：第一部分机器学习（十五）非监督度学习-2 UnsupervisedLearning-2 NeighborEmbedding](https://blog.csdn.net/dukuku5038/article/details/82949575)  
17.[机器学习与深度学习系列连载：第一部分机器学习（十六）非监督度学习-3 UnsupervisedLearning-3 Auto-Encoder ](https://blog.csdn.net/dukuku5038/article/details/82949642)  
18.[机器学习与深度学习系列连载：第一部分机器学习（十七）非监督度学习-4 UnsupervisedLearning-4 GenerativeModels ](https://blog.csdn.net/dukuku5038/article/details/82950014)  
19.[机器学习与深度学习系列连载：第一部分机器学习（十八）模型评估](https://blog.csdn.net/dukuku5038/article/details/82954769)  

-   **第一部分：深度学习（已经完成）**
     01.[机器学习与深度学习系列连载： 第二部分 深度学习(一）神经网络](https://blog.csdn.net/dukuku5038/article/details/83217542)                      
  02.[机器学习与深度学习系列连载： 第二部分 深度学习(二）梯度下降](https://blog.csdn.net/dukuku5038/article/details/83608873)                      
  03.[机器学习与深度学习系列连载： 第二部分 深度学习（三）反向传播 Backpropagaton](https://blog.csdn.net/dukuku5038/article/details/83573248)       
  04.[机器学习与深度学习系列连载： 第二部分 深度学习（四）深度学习技巧1（Deep learning tips- RMSProp + Momentum=Adam）](https://blog.csdn.net/dukuku5038/article/details/83680923)
  05.[机器学习与深度学习系列连载： 第二部分 深度学习（五）深度学习技巧2（Deep learning tips- Relu）](https://blog.csdn.net/dukuku5038/article/details/83643378)
  06.[机器学习与深度学习系列连载： 第二部分 深度学习（六）深度学习技巧3（Deep learning tips- Early stopping and Regularization）](https://blog.csdn.net/dukuku5038/article/details/83682899)
  07.[机器学习与深度学习系列连载： 第二部分 深度学习（七）深度学习技巧4（Deep learning tips- Dropout）](https://blog.csdn.net/dukuku5038/article/details/83713218)
  08.[机器学习与深度学习系列连载： 第二部分 深度学习（八）可以自己学习的激活函数（Maxout）](https://blog.csdn.net/dukuku5038/article/details/83715627)       
  09.[机器学习与深度学习系列连载： 第二部分 深度学习（九）Keras- “hello world” of deep learning](https://blog.csdn.net/dukuku5038/article/details/83721330)
  10.[机器学习与深度学习系列连载： 第二部分 深度学习（十）卷积神经网络 1 Convolutional Neural Networks ](https://blog.csdn.net/dukuku5038/article/details/83735926)
    11.[机器学习与深度学习系列连载： 第二部分 深度学习（十一）卷积神经网络 2 Why CNN for Image？](https://blog.csdn.net/dukuku5038/article/details/83774169)
  12.[机器学习与深度学习系列连载： 第二部分 深度学习（十二）卷积神经网络 3 经典的模型（LeNet-5，AlexNet ，VGGNet，GoogLeNet，ResNet）](https://blog.csdn.net/dukuku5038/article/details/83817973)
  13.[机器学习与深度学习系列连载： 第二部分 深度学习（十三）循环神经网络 1（Recurre Neural Network 基本概念 ）](https://blog.csdn.net/dukuku5038/article/details/83830994)
  14.[机器学习与深度学习系列连载： 第二部分 深度学习（十四）循环神经网络 2（Gated RNN - LSTM ）](https://blog.csdn.net/dukuku5038/article/details/83870172)
  15.[机器学习与深度学习系列连载： 第二部分 深度学习（十五）循环神经网络 3（Gated RNN - GRU）](https://blog.csdn.net/dukuku5038/article/details/83892471)
  16.[机器学习与深度学习系列连载： 第二部分 深度学习（十六）循环神经网络 4（BiDirectional RNN， Highway network， Grid-LSTM）](https://blog.csdn.net/dukuku5038/article/details/83960492)
  17.[机器学习与深度学习系列连载： 第二部分 深度学习（十七）深度神经网络调参之道（learn to learn）](https://blog.csdn.net/dukuku5038/article/details/83979866)
  18.[机器学习与深度学习系列连载： 第二部分 深度学习（十八) Seq2Seq 模型 ](https://blog.csdn.net/dukuku5038/article/details/84023100)             
  19.[机器学习与深度学习系列连载： 第二部分 深度学习（十九) 注意力机制 Attention ](https://blog.csdn.net/dukuku5038/article/details/84023470)      
  20.[机器学习与深度学习系列连载： 第二部分 深度学习（二十) 轮询采样 Scheduled Sampling](https://blog.csdn.net/dukuku5038/article/details/84060969) 
  21.[机器学习与深度学习系列连载： 第二部分 深度学习（二十一) Beam Search  ](https://blog.csdn.net/dukuku5038/article/details/84097856)          
  22.[机器学习与深度学习系列连载： 第二部分 深度学习（二十二) 机器记忆 Machine Memory ](https://blog.csdn.net/dukuku5038/article/details/84098222)  
  23.[机器学习与深度学习系列连载： 第二部分 深度学习（二十三) 空间转换层 Spatial Transfer Layer](https://blog.csdn.net/dukuku5038/article/details/84112022)
  24.[机器学习与深度学习系列连载： 第二部分 深度学习（二十四) Pointer Network  ](https://blog.csdn.net/dukuku5038/article/details/84112072)      
  25.[机器学习与深度学习系列连载： 第二部分 深度学习（二十五) 递归神经网络Resursive Network](https://blog.csdn.net/dukuku5038/article/details/84112094)
 -  **第三部分：强化学习**（待完成）
 5. 马尔科夫过程（MDP）
 6. 有模型估计（Model Predict）
 7. 无模型控制  (Model Free Control)
 8. 策略梯度（Policy Gradient)
 9. 值函数估计（Value Function Approximation )
 10. 深度强化学习 （Deep  Reinforement learning） 
 -  **第四部分：对抗学习** （待完成）
 1.对抗网络理论（Theory）
 2.条件对抗网络（Conditional GAN） 
 3.非监督条件对抗网络（Unsupervised Conditional GAN）
 4.最新的对抗网络（WGAN, EBGAN, InfoGAN, VAE-GAN, BiGAN ）
 5.对抗网络应用（Application)
 - **第五部分：深度学习框架**（待完成）
 1. TensorFlow
 2. Pytorch  
