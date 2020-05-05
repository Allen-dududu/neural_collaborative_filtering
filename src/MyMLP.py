import scipy.sparse as sp
import numpy as np
import pandas as pd
from keras import initializers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.layers import Embedding, Input, Dense, Reshape, Flatten,Dot,Concatenate
# from keras.layers import Merge
# import MyDataSet
from MyDataSet import *
import keras
from time import time
from MyEvaluate import evaluate_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    return initializers.normal(shape, scale=0.01, name=name)

# reg_layers 每一层的正则  
# layers 隐含层
def get_model(num_users, num_items, layers = [60,30], reg_layers=[0,0]):


    # # k = 128
    # model1 = Sequential()
    # model1.add(Embedding(num_users + 1, layers[0], input_length = 1))
    # model1.add(Reshape((layers[0],)))


    # model2 = Sequential()
    # model2.add(Embedding(num_items + 1, layers[0], input_length = 1))
    # model2.add(Reshape((layers[0],)))


    # # user_latent = Flatten()(model1)
    # # item_latent = Flatten()(model2)
    # # 合并层
    # Model = Sequential()
    # Model.add(keras.layers.Concatenate([model1,model2]))
    

    # Model.add(keras.layers.Dropout(0.2))
    # # 一层
    # Model.add(Dense(layers[0], activation = 'relu'))
    # Model.add(keras.layers.Dropout(0.5))
    # # 二层
    # Model.add(Dense(layers[1], activation = 'relu'))
    # Model.add(keras.layers.Dropout(0.5))
    # # 三层
    # Model.add(Dense(layers[2], activation = 'relu'))
    # Model.add(keras.layers.Dropout(0.5))

    # # 预测层
    # Model.add(Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction'))


    # return Model

    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    # output_dim 向量的维度
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = 5 , name = 'user_embedding',
                                   embeddings_regularizer = l2(0.0), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = 5, name = 'item_embedding',
                                   embeddings_regularizer = l2(0.0), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    

    vector = Concatenate()([user_latent, item_latent])
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    return model   

if __name__ == '__main__':
    dataset = MyDataSet()
    train, testRatings,testNegatives = dataset.trainMatrix, dataset.testRatings,dataset.testNegatives
    num_users, num_items = train.shape
    # 学习速率
    learning_rate = 0.001
    K = 10 # latent dimensionality
    # mu = train.rating.mean()
    epochs = 20
    batch_size = 256

    # 隐含向量
    # num_factors  = 8

    # 消极例子
    num_negatives = 4
    # # 负反馈
    # testNegatives = [] 

    # 模型输出文件名
    model_out_file = 'Pretrain/%s_MLP_%d_%d.h5'
    # 建立模型
    model = get_model(num_users, num_items)
    # 我们用普通SGD而不是Adam进行优化。 这是因为Adam需要保存更新参数的动量信息（momentum information）
    model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    t1 = time()
    testRatings = testRatings[:49]
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, 6, 1)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train,num_negatives)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        # h= hist.history['loss'][0]
        # print(str(h))
        t2 = time()
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, 5, 1)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
            % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    print("The best GMF model is saved to %s" %(model_out_file))   
