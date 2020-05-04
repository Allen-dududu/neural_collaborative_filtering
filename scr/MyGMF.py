import scipy.sparse as sp
import numpy as np
import pandas as pd
from keras import Model
import keras.backend as K
from keras.layers import Embedding, Input, Dense, Reshape, Flatten,Dot
from keras import initializers
from keras import regularizers
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
# import MyDataSet
from MyDataSet import *
import keras
from time import time
"""[summary]
返回训练一位数组
num_negatives 是负反馈样本数
Returns
-------
[type]
    [description]
"""''
def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    # num_users = train.shape[0]
    # num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while train.__contains__((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def init_normal(shape, name=None):
    # scale: 缩放因子（正浮点数）
    # shape 没有传就是使用默认的
    return initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

def get_model(num_users, num_items, latent_dim):

    # shape: 一个尺寸元组（整数），不包含批量大小
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    # output_dim 向量的维度
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim , name = 'user_embedding',
                                   embeddings_regularizer = regularizers.l2(0.0), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                   embeddings_regularizer = regularizers.l2(0.0), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    

    predict_vector = keras.layers.Multiply()([user_latent, item_latent]) 
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(predict_vector)
    
    model = Model(input=[user_input, item_input], 
                output=prediction)

    return model
if __name__ == '__main__':
    dataset = MyDataSet()
    train, testRatings = dataset.trainMatrix, dataset.testRatings
    num_users, num_items = train.shape
    # 学习速率
    learning_rate = 0.001
    K = 10 # latent dimensionality
    # mu = train.rating.mean()
    epochs = 20
    batch_size = 256
    # 隐含向量
    num_factors  = 8
    # 消极例子
    num_negatives = 8
    # 模型输出文件名
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5'
    # 建立模型
    model = get_model(num_users, num_items, num_factors)
    # 我们用普通SGD而不是Adam进行优化。 这是因为Adam需要保存更新参数的动量信息（momentum information）
    model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train,num_negatives)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        h= hist.history['loss'][0]
        print(str(h))
