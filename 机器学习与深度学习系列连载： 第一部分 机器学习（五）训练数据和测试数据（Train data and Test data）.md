#训练数据和测试数据

我们现在已经对**机器学习三板斧**已经有了比较深入的了解，其实机器学习的过程就是找到一个数学模型（函数），来进行问题求解。但是如何从找到的函数集合中挑选最好的，很多同学已经可以脱口而出了：那就是找到让Loss函数最小的函数最小就可以了。 **但是，这个让Loss函数最小的结果从哪里得出？**，这就带出来训练数据集合测试数据集的概念了。 直觉上，我们的模型在训练数据集表现的好，在测试数据集上依旧稳定，那么我们就找到了一个好的函数。但是（ 又用一个但是），我们的模型选择并不是这么简单，是一个相对复杂的过程。（因为还有超参数设置，比如学习率）
那我们就举一个简单的例子：
![这里写图片描述](https://img-blog.csdn.net/20180914104652345?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
模型3在训练集合中表现最好，一般我们选用模型三进行测试。但是如果我们把测试数据分为，两个部分，一个是公共测试数据，一个是私有测试数据。
![这里写图片描述](https://img-blog.csdn.net/20180914104934105?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
那就是公共测试数据中模型3表现最好，但是在私有测试数据中表现不理想。 这个问题怎么破
##交叉验证 Cross Validation
首先我们把训练数据拆分成两部分，一部分是用户训练，一部分用于验证测试。验证测试的结果帮我我们进行模型选择。（函数选择和参数调整）
![这里写图片描述](https://img-blog.csdn.net/20180914105359478?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)	
然后将选出来的模型，在所有的训练数据再次训练后，再到公共的测试数据中进行测试

**为了是数据利用的更加充分，我们选择N阶交叉验证**
###N Fold Cross Validation
![这里写图片描述](https://img-blog.csdn.net/20180914105758945?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1a3VrdTUwMzg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
图例中N=3,我们从3次交叉验证中，得出，Model1 在训练中的表现最好，然后在用户公共测试。
这样可以在正式测试前，将模型尽可能调整到最优状态。
