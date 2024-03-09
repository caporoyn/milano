#!/usr/bin/env python
# coding: utf-8

# # **資料準備**

# In[1]:



#   分 train:50+1       test: 10+1
#   以每日144時段預測明日巔峰時段位置/區域

#導入python操作系統模組
#保證程式中的GPU序號是和硬體中的序號相同的
#使用第一張GPU卡
import os  
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import keras
import tensorflow as tf
from keras.datasets import cifar10
from keras import backend as K
#導入後端模組K=Theano or TensorFlow 
import time
#執行時間
from keras.layers import Input, Conv2D,GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout, Reshape, Permute, multiply
#啟動
from keras.layers import  ReLU, Add,Softmax,Flatten
from keras.models import Model
#函数式API
from keras.layers import DepthwiseConv2D, MaxPooling2D, AveragePooling2D, concatenate
#from se_block import squeeze_excite_block

from keras import optimizers,regularizers
#優化器,正規化
from keras.preprocessing.image import ImageDataGenerator
#資料增強
from keras.regularizers import l2
from keras.initializers import he_normal
#初始化器(設定 Keras各層權重隨機初始值的方法)he_normal:均勻分佈抽取樣本
from sklearn.model_selection import train_test_split
#分離器函式:用於將陣列或矩陣劃分為訓練集和測試集
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
#callbacks回調函數,學習速率定時器.基本可視化.在每個訓練週期之後保存模型
import milan


num_classes        = 5
batch_size         = 64         # 64 or 32 or other
epochs             = 100
iterations         = 39
#每一次迭代都是一次權重更新
USE_BN=True
DROPOUT=0.2 # keep 80%
CONCAT_AXIS=3
#concat_axis表示特徵軸，因爲連接和BN都是對特徵軸而言的
weight_decay=1e-4
#權重衰減
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
#字符串,代表圖像的通道的位置
log_filepath  = './SE-Densenet'
#系統變數,指令歷程記錄檔的路徑
import numpy as np
#引入 numpy 模組
np.random.seed(5)
#可以保證生成的隨機數具有可預測性
import datetime

# # **資料預處理並設置learning schedule**

# In[2]:


IM_WIDTH, IM_HEIGHT = 100, 100

train_dir = '/Users//xieyourong/Desktop/mnlab/milano/dataset/train'  # 訓練集數據
val_dir = '/Users//xieyourong/Desktop/mnlab/milano/dataset//test' # 驗證集數據
input_depth = 144


# In[3]:


#資料預處理並設定 learning schedule
#x_train_normalize = x_train.astype('float32') / 255.0
#x_test_normalize = x_test.astype('float32') / 255.0
#x_val_normalize = x_val.astype('float32') / 255.0

#def color_preprocessing(x_train,x_test):    
    #均值
#    mean = [125.307, 122.95, 113.865]
    #張量的標準差
#    std  = [62.9932, 62.0887, 66.7048]    
    #標準化-data preprocessing  [raw - mean / std]
#    for i in range(3):
#        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
#        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
#    return x_train, x_test
    
    
#Learning Rate Schedule(學習速率表)根迭代次數改變scheduler，越迭代到後面該值越小，這意味著希望訓練過程中隨機因素逐步減小：
def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 200:
        return 0.001
    return 0.0001


# In[4]:
date = datetime.date(2013,11,1)
(x_train , y_train) , (x_test , y_test) = milan.load_data(date)




# In[5]:



# # **定義網路結構**

# In[6]:


def Depthwise_block(x, kernel=(3, 3), strides=(1, 1), activation='RE', nb_layers=0):
    x = DepthwiseConv2D((3,3),strides=(1,1), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
   
    if activation == 'RE':
        x = ReLU()(x)
    return x


# In[7]:


# DenseNet Block：每一層Block裡，是由一層1x1conv與3x3conv所組成
def Conv_Block(input_tensor, filters, bottleneck=True, dropout_rate=None, weight_decay=1e-4):
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1  # 確定格式
    
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_tensor) #將張量傳入層函數，然後回傳結果張量
    x = Activation('relu')(x)

    if bottleneck:
        # 使用bottleneck進行降維** pointwise conv
        inter_channel = filters * 4
        x = Conv2D(inter_channel, (1, 1),kernel_initializer='he_normal',padding='same', use_bias=False,
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)    
         
    x = Depthwise_block(x, kernel=(3, 3),strides=(1, 1),activation='RE')
#x = depthwise_separable(x,params=[(1,),(64,)])
    x = Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x) 
#test5-2 加入BN+ReLU

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


# In[8]:


# DenseNet-Transition Block:一層1x1conv、一層2x2pool，步長2縮減圖像大小
def Transition_Block(input_tensor, filters, compression_rate, weight_decay=1e-4):

    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1  # 确定格式

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(int(filters * compression_rate), (1, 1),
              kernel_initializer='he_normal',
              padding='same',
              use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


# In[9]:


# DenseNet Block(用於串接層與層之間的權重)
def Dense_Block(x, nb_layers, filters, growth_rate, bottleneck=True, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True, return_concat_list=False):

    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1
    x_list = [x]

    for i in range(nb_layers):
        cb = Conv_Block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)
        #將一個新的項目加到 list 的尾端
        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            filters += growth_rate

    if return_concat_list:
        return x, filters, x_list
    else:
        return x, filters


# In[10]:


# SENet Block
def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channel_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channel_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


# # **搭建網路**

# In[11]:


def DenseMobileNet (classes=5, input_shape=(100, 100, 1), include_top=True, nb_dense_block=4, growth_rate=64, nb_filter=64,
            nb_layers_per_block=[3, 3, 3, 3], bottleneck=True, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4,
            subsample_initial_block=True):

        #nb_filter=濾波器個數？每層數量固定
    #nb_dense_block 層數
    #nb_layers_per_block 每層block的執行次數
    
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1

    final_nb_layer = nb_layers_per_block[-1] # 輸出最後一個字符 即 9
    nb_layers = nb_layers_per_block[:-1] # 正向輸出  從開始 ~ 倒數第第1個字符（不含第1個）即 012345678

    compression = 1.0 - reduction #reduction減少=0.5？
    if subsample_initial_block:
        initial_kernel = (3, 3) #初始卷積核大小#
        initial_strides = (1, 1)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)
    input_tensor = Input(shape=input_shape)    
    
    #第一層：標準卷積
    x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
              strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(input_tensor) #weight_decay:權重衰退

    #python：if...for迴圈：
    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x) #epsilon：浮點數，用於避免在某些操作中被零除的數字模糊常量。
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x) #第一個pooling原設為3X3，改為2X2#

    for block_index in range(nb_dense_block - 1): #dense_block層數（設為3）-迴圈重複層數                
        
        #使用Dense Block
        x, nb_filter = Dense_Block(x, nb_layers[block_index], nb_filter, growth_rate,
                                   bottleneck=bottleneck,dropout_rate=dropout_rate, weight_decay=weight_decay)         
        
        #使用Depthwise Block的可分離（新增）
#        x, nb_filter = Depthwise_block(x, nb_layers[block_index], nb_filter, growth_rate, 
#                                   bottleneck=bottleneck,dropout_rate=dropout_rate, weight_decay=weight_decay)                 
        
        #使用SE Block (新增)
        x = squeeze_excite_block(x)      
        
        #使用Transition Block 
        x = Transition_Block(x, nb_filter, compression_rate=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
                        
    #最後一層使用Dense Block    
    x, nb_filter = Dense_Block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                              dropout_rate=dropout_rate, weight_decay=weight_decay)    
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:  #include_top：是否保留頂層的全連接層
        x = Dense(classes, activation='softmax')(x)

    model = Model(input_tensor, x, name='densenet121')

    return model


# In[12]:


model = DenseMobileNet(include_top=True)
model.summary()


# # **生成模型**

# In[13]:


try:
    model.load_weights('/Users/xieyourong/Desktop/mnlab/milano/milano.h5')
    print("載入模型成功!繼續訓練模型")
except :    
    print("載入模型失敗!開始訓練一個新模型")


# # **開始訓練**

# In[13]:


train_samples = 932
validation_samples = 235


# In[14]:


# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#opt = keras.optimizers.rmsprop(lr=0.1, decay=1e-4)
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# set callback
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]

# set data augmentation
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)
#datagen.fit(x_train_normalize)
start = time.time()
# start training
#train_history = model.fit_generator(datagen.flow(x_train_normalize, y_train_OneHot,
#                    batch_size=batch_size),
#                    steps_per_epoch=iterations,
#                    epochs=epochs,
#                    callbacks=cbks,
#                    validation_data=(x_val_normalize, y_val_OneHot))


train_history = model.fit_generator(
#    datagen.flow(x_train_normalize, y_train_OneHot,
                    train_generator,
                   # batch_size=batch_size,
                    steps_per_epoch=np.floor(train_generator.n/batch_size),
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=val_generator,
                    validation_steps=validation_samples// batch_size)



model.save('mobilenet.h5')
end = time.time()


# In[15]:


print ("Model took %0.2f seconds to train" % (end - start))


# In[16]:


import matplotlib.pyplot as plt
def show_train_history(train ,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.savefig('1.png')
    plt.show()


# In[17]:


show_train_history('acc','val_acc')


# In[18]:


show_train_history('loss','val_loss')


# In[19]:


model.evaluate_generator(val_generator, validation_samples)


# In[ ]:


#score = model.evaluate(x_test_normalize,y_test_OneHot, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#predictions = model.predict(x_test_normalize, verbose=1)

