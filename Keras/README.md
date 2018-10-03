# deepLeaning_platfrom
1、Keras 提取网络参数 <br>
```
weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]
```
2、keras 计算top5、top3

```
def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy', top_k_categorical_accuracy, acc_top3])
```
其中top_k_categorical_accuracy，默认是top5。

## 分类参考链接
[baidu_dog](https://github.com/ahangchen/keras-dogs)
