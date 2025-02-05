# Learning Resources

- [course website](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/)
- [course vedio](https://www.youtube.com/watch?v=dJYGatp4SvA&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=1)

### HW1 - KNN

- [K-Nearest Neighbor(KNN) Algorithm](https://www.geeksforgeeks.org/k-nearest-neighbours/)

### HW2 - Linear Classification & 2-layer Network

- [03-linear_classification_slides](./Slides/03-Linear_Classification.pdf)

- [cs231n-reference-book-SVM](https://sharad-s.gitbooks.io/cs231n/content/lecture_3_-_loss_functions_and_optimization/multiclass_svm_loss_deep_dive.html)

- [04-optimization-ppt](./Slides/04-optimization.pdf)

- [cs231n-reference-book-Optimization](https://sharad-s.gitbooks.io/cs231n/content/lecture_3_-_loss_functions_and_optimization/optimization_-_gradient_descent.html)

- [cs231n-softmax_loss](https://cs231n.github.io/linear-classify/#softmax-classifier)

### HW3 - Conv_RNN

- [06-backpropagation-ppt](./Slides/06-backpropagation.pdf)
- [Conv_backpropagation](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)
- [MaxPool_backward](https://www.educative.io/answers/how-to-backpropagate-through-max-pooling-layers)
- [learning_rate_update](https://cs231n.github.io/neural-networks-3/#sgd)

##### weight_initialization

- [kaiming_weight_initialization](https://pouannes.github.io/blog/initialization/)

  | **初始化方法** | **适用的激活函数**               | **优点**                                            | **缺点**                                   |
  | -------------- | -------------------------------- | --------------------------------------------------- | ------------------------------------------ |
  | **Kaiming**    | ReLU、Leaky ReLU、ELU 等激活函数 | 适用于 ReLU 激活函数，能有效避免梯度消失问题        | 不适合 Sigmoid、Tanh 等激活函数            |
  | **Xavier**     | Sigmoid、Tanh 等激活函数         | 适用于 Sigmoid/Tanh，防止梯度消失问题，适合浅层网络 | 对 ReLU 激活函数表现较差，深层网络效果较差 |

##### Batch Normalization

- [ConvNet_Slides](./Slides/07-ConvNet.pdf)
- Makes deep networks much eaiser to train
- Allows higher learning rates, faster convergence
- Networks become more robust to initialization
- Acts as regularization during training
- Zero overhead at test-time: can be fused with conv!

---

- Not well-understood theoretically (yet)
- Behaves differently during training and testing: this is a very common source of bugs!
