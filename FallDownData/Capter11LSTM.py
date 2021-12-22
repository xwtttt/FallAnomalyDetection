'''
Description: 
Author: liguang-ops
Github: https://github.com/liguang-ops
Date: 2020-12-11 11:57:41
LastEditors: liguang-ops
LastEditTime: 2020-12-14 21:43:16
'''
import numpy as np 
import tensorflow as tf
from tensorflow.keras import Sequential,losses,layers,datasets,optimizers
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        #设置GPU显存占用为按需分配方式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
            print(e)

batchsz = 128          #batchsize
total_words = 10000    #词汇表大小
max_review_len = 80    #句子最大长度
embedding_len = 100    #词向量长度

(x_train,y_train),(x_test,y_test) = datasets.imdb.load_data(num_words=total_words)

#将字典的value后移，保留0-3位，塞入新的词索引
word_index = datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}    
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3

#填充短句子，默认填充在首部
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_len)

#构建训练和测试的数据集
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(1000). \
                                                                batch(batchsz,drop_remainder=True)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(1000). \
                                                                batch(batchsz,drop_remainder=True)


class LSTMlayer(layers.Layer):
    def __init__(self,sec_dim,units):
        super(LSTMlayer,self).__init__()
        self.sec_dim = sec_dim   #输入训练数据的第二维
        self.units = units       #状态矩阵 
        #遗忘门的权重
        self._wf = self.init_weights()
        self._bf = self.init_bais()

        #更新门权重
        self._wr1 = self.init_weights()
        self._br1 = self.init_bais()

        self._wr2 = self.init_weights()
        self._br2 = self.init_bais()

        #输出门权重
        self._wo = self.init_weights()
        self._bo = self.init_bais()
    
    def init_weights(self):
        #初始化权重  [units,units + n(sec_num)]
        return tf.Variable(tf.random.normal([self.units + self.sec_dim,self.units],stddev=0.1))
    
    def init_bais(self):
        #初始化偏置  [units,]
        return tf.Variable(tf.zeros([self.units]))

    def call(self,inputs,chlist,training = None):
        #param inputs [batchsz,n],n可以理解为词向量长度 n = sec_dim ,b = batchsz
        #param chlist 保存上一时间戳的c和h
        c_before = chlist[0]     #上一时刻的c [b,units]
        h_before = chlist[1]     #上一时刻的h [b,units]
        x = inputs
        #遗忘门
        #[b,units] = [b,n+units] @ [n+units,units] + [units,]
        gf = tf.nn.sigmoid(tf.concat([h_before,x],axis=1) @ self._wf + self._bf)
        #遗忘门输出 gf * c_before

        #输入门(更新)
        gr = tf.nn.sigmoid(tf.concat([h_before,x],axis=1) @ self._wr1 + self._br1)
        #临时变量
        tem = tf.concat([h_before,x],axis=1) @ self._wr2 + self._br2
        #C～t,临时变量
        c_tem = tf.tanh(tem)
        #ct输出
        c_current = gf * c_before + gr * c_tem
        
        #输出门
        go = tf.sigmoid(tf.concat([h_before,x],axis =1) @ self._wo + self._bo)
        h_current = go * tf.tanh(c_current)
        return h_current,[c_current,h_current]

class Model(keras.Model):
    def __init__(self,batchsz,units):
        super(Model, self).__init__()
        #加载词向量模型
        embeddings_index = {}
        with open('./glove/glove.6B.100d.txt','r',encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:],dtype='float32')
                embeddings_index[word] = coefs
        num_words = min(total_words,len(word_index))
        embedding_matrix = np.zeros((num_words,embedding_len))
        for word,i in word_index.items():
            if i>=total_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
        #lstm层初始状态lstmich list = [[b,units],[b,units]]
        self.lstm1ch = [tf.zeros([batchsz,units]),tf.zeros([batchsz,units])]     
        self.lstm2ch = [tf.zeros([batchsz,units]),tf.zeros([batchsz,units])]

        #embedding层
        self.embedding = layers.Embedding(total_words,embedding_len,input_length=max_review_len,
                                                                                trainable = False)
        #input_shape = [b,80] => [b,80,100]
        self.embedding.build(input_shape = (None,max_review_len))
        #设置embedding层参数
        self.embedding.set_weights([embedding_matrix])
        self.lstm1 = LSTMlayer(embedding_len,units)
        self.lstm2 = LSTMlayer(units,units)
        #分类
        self.outlayer = layers.Dense(1)
    
    def call(self,inputs,training=None):
        x = inputs  #[b,80]
        #[b,80] => [b,80,100]
        x = self.embedding(x)
        #两个lstm层的初始状态
        state0 = self.lstm1ch
        state1 = self.lstm2ch 
        
        for xt in tf.unstack(x,axis=1):
            #第一层lstm，输入为[b,n],n为sec_dim参数
            #到第二个lstm，输入为[b,units],那么第二层的sec_dim参数应该为units
            #[b,80,100] => [b,100] =>[b,units]
            out0,state0  = self.lstm1(xt,state0,training)
            #print(out0.shape,state0[0].shape,state0[1].shape)
            #[b,units] => [b,units]
            out1,state1 = self.lstm2(out0,state1,training)
        #[b,units] => [b,1]
        x = self.outlayer(out1)
        prob = tf.sigmoid(x)
        return prob

        
units = 64
epochs = 20

lstm = Model(batchsz,units)
lstm.compile(optimizer = optimizers.Adam(0.001),
                                    loss=losses.BinaryCrossentropy(),metrics=['accuracy'])

lstm.fit(train_db,epochs = epochs,validation_data=test_db)


        