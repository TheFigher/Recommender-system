"""
keras 实现FM算法
"""
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
K = tf.keras.backend


#  自定义FM的二阶交叉层
class FMLayer(tf.keras.layers.Layer):
    # FM的K取4
    def __init__(self,input_dim, output_dim=4,**kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(FMLayer,self).__init__(**kwargs)

    # 初始化训练权重
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernrl',# F*K
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(FMLayer, self).build(input_shape)

    # 自定义FM的二阶交叉项计算公式
    def call(self, x): # X维度B*F
        a = K.pow(K.dot(x, self.kernel), 2) # B*K
        b = K.dot(K.pow(x,2),K.pow(self.kernel, 2))  #B*K
        return K.sum(a-b,1,keepdims=True) * 0.5

    # 输出尺寸大小
    def compute_output_shape(self, input_shape):
        return input_shape[0], 1

# 实现FM算法
def FM(feature_dim):
    inputs = tf.keras.Input((feature_dim,))  # B*F
    # 线性回归
    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.02))(inputs) # B*1
    # FM的二阶交叉项
    cross = FMLayer(feature_dim)(inputs) # B*1

    # 获得FM模型 （线性回归 + FM的二阶交叉项）
    add = tf.keras.layers.Add()([liner, cross])  # B*1
    predictions = tf.keras.layers.Activation('sigmoid')(add) # B*1

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.binary_accuracy])
    return model


# 训练FM模型
def train():
    fm = FM(30)
    data = load_breast_cancer()

    # sklearn 切分数据
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=11, stratify=data.target)
    fm.fit(X_train, y_train, epochs=5, batch_size=20, validation_data=(X_test, y_test))
    return fm


if __name__ == '__main__':
    fm = train()
    fm.summary()




# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import keras
# import os
#
# import matplotlib.pyplot as plt
#
# from keras.layers import Layer,Dense,Dropout,Input
# from keras import Model,activations
# from keras.optimizers import Adam
# from keras import backend as K
# from keras.layers import Layer
# from sklearn.datasets import load_breast_cancer
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# class FM(Layer):
#     def __init__(self, output_dim, latent=10,  activation='relu', **kwargs):
#         self.latent = latent
#         self.output_dim = output_dim
#         self.activation = activations.get(activation)
#         super(FM, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.b = self.add_weight(name='W0',
#                                   shape=(self.output_dim,),
#                                   trainable=True,
#                                  initializer='zeros')
#         self.w = self.add_weight(name='W',
#                                  shape=(input_shape[1], self.output_dim),
#                                  trainable=True,
#                                  initializer='random_uniform')
#         self.v= self.add_weight(name='V',
#                                  shape=(input_shape[1], self.latent),
#                                  trainable=True,
#                                 initializer='random_uniform')
#         super(FM, self).build(input_shape)
#
#     def call(self, inputs, **kwargs):
#         x = inputs
#         x_square = K.square(x)
#
#         xv = K.square(K.dot(x, self.v))
#         xw = K.dot(x, self.w)
#
#         p = 0.5*K.sum(xv-K.dot(x_square, K.square(self.v)), 1)
#
#         rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)
#
#         f = xw + rp + self.b
#
#         output = K.reshape(f, (-1, self.output_dim))
#
#         return output
#
#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape)==2
#         return input_shape[0],self.output_dim
#
#
# data = load_breast_cancer()["data"]
# target = load_breast_cancer()["target"]
#
# K.clear_session()
# print(target)
# inputs = Input(shape=(30,))
# out = FM(20)(inputs)
# out = Dense(15, activation='sigmoid')(out)
# out = Dense(1, activation='sigmoid')(out)
#
# model=Model(inputs=inputs, outputs=out)
# model.compile(loss='mse',
#               optimizer='adam',
#               metrics=['acc'])
# model.summary()
#
# h=model.fit(data, target, batch_size=1, epochs=10, validation_split=0.2)
#
# #%%
#
# plt.plot(h.history['acc'],label='acc')
# plt.plot(h.history['val_acc'],label='val_acc')
# plt.xlabel('epoch')
# plt.ylabel('acc')
#
# #%%
