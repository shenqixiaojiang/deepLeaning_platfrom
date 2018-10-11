import keras
import itertools
import numpy as np

from keras import backend as K 
from keras.models import Model
from keras.layers import Input, Dense, Activation

#from ipdb import set_trace as bp

class WeightedCategoricalCrossEntropy(object):

  def __init__(self, weights):
    nb_cl = len(weights)
    self.weights = np.ones((nb_cl, nb_cl))
    for class_idx, class_weight in weights.items():
        self.weights[0][class_idx] = class_weight
        self.weights[class_idx][0] = class_weight
    self.__name__ = 'w_categorical_crossentropy'

  def __call__(self, y_true, y_pred):
    return self.w_categorical_crossentropy(y_true, y_pred)

  def w_categorical_crossentropy(self, y_true, y_pred):
    nb_cl = len(self.weights)
    final_mask = K.zeros_like(y_pred[..., 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max, axis=-1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        w = K.cast(self.weights[c_t, c_p], K.floatx())
        y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
        y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
        final_mask += w * y_p * y_t
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
# create a toy model
i = Input(shape=(100,))
h = Dense(7)(i)
o = Activation('softmax')(h)

model = Model(inputs=i, outputs=o)


# compile the model with custom loss
loss = WeightedCategoricalCrossEntropy({0: 1.0, 1: 29.6, 2: 17.69, 3: 27.08, 4: 11.04, 5: 45.45, 6: 136.344})
model.compile(loss=loss, optimizer='sgd')
print "Compilation OK!"

# fit model
model.fit(np.random.random((64, 100)),np.random.random((64, 7)), epochs=10)

# save and load model
model.save('model.h5')
model = keras.models.load_model('model.h5', custom_objects={'w_categorical_crossentropy': loss})
print "Load OK!"
