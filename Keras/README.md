# deepLeaning_platfrom
1、Keras 提取网络参数 <br>
```
weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]
```
