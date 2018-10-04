# deepLeaning_platfrom
1、Keras 提取网络参数 <br>
```
weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]
```
2、keras 计算top5、top3

```
def top3_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy', top_k_categorical_accuracy, top3_acc])
```
其中top_k_categorical_accuracy，默认是top5。<br>
* 当load model时，会报错 unkonwn metrics top3_acc，解决方法：
```
model = load_model('saved_models/' +  model_name + '.h5',custom_objects={'top3_acc': top3_acc})
```

3、keras 计算F1
```
from keras import backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
```

4、Keras 图像增强[源码](https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py)

## 分类参考链接
[baidu_dog](https://github.com/ahangchen/keras-dogs)
