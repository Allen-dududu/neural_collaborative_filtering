# neural_collaborative_filtering
神经网络协同过滤论文复现</br>
论文地址 https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf</br>
<span>数据集在<a>https://github.com/Alean-singleDog/Recommend-system-HomeWork</a>项目里，我就不重复上传了</span>
<h3>摘要</h3>
<span>近年来，深层神经网络在语音识别，计算机视觉和自然语言处理方面都取得了巨大的成功。然而相对的，对应用深层神经网络的推荐系统的探索却受到较少的关注。在这项工作中，我们力求开发一种基于神经网络的技术，来解决在含有隐形反馈的基础上进行推荐的关键问题————协同过滤。
　　尽管最近的一些工作已经把深度学习运用到了推荐中，但是他们主要是用它（深度学习）来对一些辅助信息（auxiliary information）建模，比如描述文字的项目和音乐的声学特征。当涉及到建模协同过滤的关键因素（key factor）————用户和项目（item）特征之间的交互的时候，他们仍然采用矩阵分解的方式，并将内积（inner product）做为用户和项目的潜在特征点乘。通过用神经结构代替内积这可以从数据中学习任意函数，据此我们提出一种通用框架，我们称它为NCF（Neural network-based Collaborative Filtering，基于神经网络的协同过滤）。NCF是一种通用的框架，它可以表达和推广矩阵分解。为了提升NFC的非线性建模能力，我们提出了使用多层感知机去学习用户-项目之间交互函数（interaction function）。在两个真实世界（real-world）数据集的广泛实验显示了我们提出的NCF框架对最先进的方法的显著改进。</span>
