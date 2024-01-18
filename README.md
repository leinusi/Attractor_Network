数字识别使用吸引子网络
项目介绍
这个项目展示了如何使用吸引子网络(Attractor Network)来识别手写数字。项目使用了 scikit-learn 的 load_digits 数据集，并实现了一个基本的吸引子网络类，该类用于训练模型并进行预测。

安装指南
为了运行此项目，您需要安装以下依赖项：

Python 3.x
NumPy
Matplotlib
scikit-learn
您可以使用 pip 安装这些依赖项：

bash
Copy code
pip install numpy matplotlib scikit-learn
使用方法
要使用这个项目，您只需运行包含吸引子网络定义和训练逻辑的脚本。该脚本将训练一个模型来识别手写数字，并在最后显示其在测试集上的准确率和混淆矩阵。

代码说明
AttractorNetwork 类：定义了吸引子网络，包括权重初始化、邻接矩阵创建、softmax 函数、细胞状态更新、训练和预测函数。
数据预处理：使用 scikit-learn 的 load_digits 加载并预处理数据。
训练和评估：定义了网络的细胞数、类别数、学习率和训练周期，然后对网络进行训练并评估其在测试集上的性能。
可视化：使用 Matplotlib 显示混淆矩阵。
许可证
MIT License
