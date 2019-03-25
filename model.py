from keras.models import *
from keras import *
from keras.utils import *
from keras.layers import *
import scipy.io as io
import scipy as cs
from keras.legacy import *
import numpy as np
from sklearn.naive_bayes import GaussianNB
from h5py import *


def get_Model():
    G = Sequential()
    G.add(Dense(16, input_dim=16, activation='relu'))

    G.add(Dense(16, activation='relu'))

    G.add(Dense(16,activation='sigmoid'))
    G.summary()

    D = Sequential()
    D.add(Dense(16, activation='relu', input_dim=16))

    D.add((Dense(16, activation='relu')))

    D.add(Dense(1))
    D.summary()

    DM = Sequential()
    DM.add(D)
    DM.compile(loss="mse", optimizer='RMSprop', metrics=['accuracy'])

    AM = Sequential()
    AM.add(G)
    AM.add(D)
    AM.compile(loss="mse", optimizer='RMSprop', metrics=['accuracy'])

    return AM, DM,G


def get_data(name,part,day):
    file_name = 'E:\ksdler\\Newdata\\Newdata'
    days = copy.deepcopy(day)
    if(days!=10):
        days = "0"+str(days)
    data = File(file_name+'\\'+name+"day"+days+"_part"+str(part)+".mat")
    return np.array(copy.deepcopy(data["trainData"][:]))

def regularzation(x):
    for i in range(len(x)):
        x[i,:] = (x[i,:] - np.min(x[i,:]))/(np.max(x[i,:])-np.min(x[i,:]))
    return x

def bayes(x,y):
    f = 0
    model = GaussianNB()
    model.fit(x,y)
    y_ = model.predict(x)
    for i in range(len(y_)):
        if(y_[i]== y[i]):
            f = f+1
    print("acc = " + str(f/len(y)))
    return model

def random(x,size = 1000):
    if(size>1550):
        size = 1550
    size = copy.deepcopy(int(np.ceil(size/10)))
    x_return = x[0:size]
    x_return = x_return[np.random.randint(0, x_return.shape[0], size=size),:]
    for i in range(1,10):
        x_temp = copy.deepcopy(x[i*155:(i+1)*155])
        x_temp = x_temp[np.random.randint(0, x_temp.shape[0], size=size),:]
        print("x_temp")
        print(x_temp)
        x_return = np.concatenate((x_return,copy.deepcopy(x_temp)))
    return x_return

def nomal_model():
    G = Sequential()
    G.add(Dense(16, input_dim=16, activation='relu',kernel_initializer=initializers.glorot_normal()))
    G.add(Dropout(0.4))

    G.add(Dense(16, activation='relu',kernel_initializer=initializers.glorot_normal()))
    G.add(Dropout(0.4))
    G.add(Dense(16, activation='relu',kernel_initializer=initializers.glorot_normal()))
    G.add(Dropout(0.4))

    G.add(Dense(16,kernel_initializer=initializers.glorot_normal()))
    G.compile(loss="mse", optimizer=optimizers.RMSprop(lr = 0.05,decay=0.0001), metrics=['accuracy'])
    return G


def get_acc(y,y_):
    f = 0
    for i in range(len(y)):
        if(int(y[i])==int(y_[i])):
            f = f+1
    return f/len(y)


def train(x,y,batch_size = 100):
    AM,DM,G= get_Model()
    for i in range(5000):
        G_train = x[np.random.randint(0, x.shape[0], size=batch_size), :]
        D_train = y[np.random.randint(0, y.shape[0], size=batch_size), :]
        print("D_train")
        print(D_train[0])
        fake = G.predict(G_train)
        print("Fake")
        print(fake[0])
        train_trueAndFake = np.concatenate((fake, D_train))
        result = np.ones((batch_size * 2, 1))
        result[:batch_size, :] = 0
        print("DM train in loop "+ str(i))

        DM.fit(train_trueAndFake, result, epochs=2)
        for l in DM.layers:
           weights = l.get_weights()
           weights = [np.clip(w, -1, 1) for w in weights]
           l.set_weights(weights)

        z = DM.predict(train_trueAndFake)
        print("fake acc = "+str(z[0]))
        print("true acc = "+str(z[-1]))
        result = np.ones((batch_size, 1))
        print("GM train in loop "+str(i))
        AM.fit(G_train, result, epochs=1)
    return G

def get_dataInLabel(x,y,label,size = 128):
    find = True
    for i in range(len(y)):
        if(y[i]==label):
            if(find == True):
                temp = np.array([copy.deepcopy(x[i,:])])
                find = False
            else:
                t = np.array([x[i,:]])
                temp = np.concatenate((temp,t))
    return np.array(temp)

def train_nomal(x,y,label_x,label_y,label,batch_size = 512):
    model = nomal_model()
    for i in range(500):
       DATA = get_dataInLabel(x,label_x,label = label,size = int(batch_size/4))
       LABEL = get_dataInLabel(y,label_y,label = label,size = int(batch_size/4))
       model.fit(DATA,LABEL,epochs=1)
       print(i)
       print("Fake")
       print(model.predict(DATA)[0])
       print("Label")
       print(LABEL[0])
    return model

def batch_train(x,y,label_x,label_y,batch_size = 512):
    models = []
    for i in range(10):
        temp1 = copy.deepcopy(x)
        temp2 = copy.deepcopy(y)
        models.append(train_nomal(copy.deepcopy(temp1),copy.deepcopy(temp2),label_x=label_x,label_y=label_y,label=i+1,batch_size=batch_size))
    return models

def get_multiAcc(models,x,y,test_x,test_y):
    len = 10
    for i in range(len):
        thi = copy.deepcopy(models[i].predict(get_dataInLabel(x,y,i+1)))
        if(i==0):
            x_ = copy.deepcopy(thi)
        else:
            x_ = np.concatenate((x_,copy.deepcopy(thi)))

    print(x_)
    print(y)
    for i in test_y:
        print(i)
    m = bayes(test_x,test_y)
    y_ = m.predict(x_)
    for i in y_:
        print(i)

    return get_acc(y_,y)



data1 = get_data("YX",day=1,part=1)
data2 = get_data("YX",day=2,part=1)
data3 = get_data("YX",day=3,part=1)
x1 = data1[:,:16]
x2 = data2[:,:16]
x3 = data3[:,:16]
y1 = data1[:,16:17]
y2 = data2[:,16:17]
y3 = data3[:,16:17]
model = bayes(x1,y1)
y2_ = model.predict(x2)
print(get_acc(y2_,y2))

yfdata1 = get_data("YF",day = 1,part = 1)
yfdata2 = get_data("YF",day = 2,part = 1)
yjdata1 = get_data("YJ",day = 1,part = 1)
yjdata2 = get_data("YJ",day = 2,part = 1)
krdata1 = get_data("KR",day = 1,part = 1)
krdata2 = get_data("KR",day = 2,part = 1)
dsdata1 = get_data("DS",day = 1,part = 1)
dsdata2 = get_data("DS",day = 2,part = 1)


d1 = np.concatenate((yfdata1[:,:16],yjdata1[:,:16],krdata1[:,:16],dsdata1[:,:16]))
d2 = np.concatenate((yfdata2[:,:16],yjdata2[:,:16],krdata2[:,:16],dsdata2[:,:16]))
l1 = np.concatenate((yfdata1[:,16:17],yjdata1[:,16:17],dsdata1[:,16:17],krdata1[:,16:17]))
l2 = np.concatenate((yfdata2[:,16:17],yjdata2[:,16:17],dsdata2[:,16:17],krdata2[:,16:17]))
print(l2)
da1 = copy.deepcopy(regularzation(yfdata2[:,:16]))
da2 = copy.deepcopy(regularzation(yfdata1[:,:16]))
la1 = copy.deepcopy(yfdata2[:,16:17])
la2 = copy.deepcopy(yfdata1[:,16:17])

models = batch_train(d2,d1,l2,l1,batch_size=512)
print(get_multiAcc(models,x2,y2,x1,y1))