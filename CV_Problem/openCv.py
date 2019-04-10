import pickle  #To open .pkl file for unpickling
import numpy as np  #To work around arrays
from keras.models import Sequential   #Keras is dl api and sequential is used to create mode layer by layer
import tensorflow as tf  #base for all processing machine learning
import csv  #To export result as CSV
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D #Convoluting,flattening the data

def mediansitetraindata(imtrdb,imtrldb,imprdb):
    imtraindb=[]  #image train data
    imtrainlabeldb=[] #image train label

    impredictiondb=[] #Test data or images

    m=0
    while(m<len(imtrdb)):
        j = np.array(imtrdb[m]).reshape(28, 28)
        imtraindb.append(j.tolist())   #As in imtraindb contains 1d array of images data,so traversing
        #it and getting one image at a time which is of 784 size then reshaping it to 28*28
        imtrainlabeldb.append(imtrldb[m]) #The label for train data
        m=m+1
    # while(m>6800 and m<8000):
    #     j = np.array(imtrdb[m]).reshape(28, 28)
    #     imtestdb.append(j.tolist())
    #     imtestlabeldb.append(imtrldb[m])
    #     m = m + 1

    m=0
    while(m<len(imprdb)):
       j = np.array(imprdb[m]).reshape(28, 28) #Test data or prediction data
       impredictiondb.append(j.tolist())
       m=m+1


    tp=np.asarray(imtraindb)  #making list to np array for processing
    pp=np.asarray(impredictiondb)
    x_train = tp.reshape(tp.shape[0], 28, 28, 1)  #The tp.shape is the size of the array in which
    # each data is of 28 rows and col with 1 ie. greyscalevalue
    x_test = pp.reshape(pp.shape[0], 28, 28, 1)

    input_shape = (28, 28,1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the  codes by dividing it to the max  value.
    x_train /= 255
    x_test /= 255
    # print('x_train shape:', x_train.shape) #Getting the values and shape
    # print('Number of images in x_train', x_train.shape[0])
    # print('Number of images in x_test', x_test.shape[0])

    model = Sequential() # creating a instance of sequential model
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape)) #Convlouting with a matrix of kernel of size 3,3
    model.add(MaxPooling2D(pool_size=(2, 2))) #To down sample the image data
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))  #Initially it can be any number. I got good value at this one
    model.add(Dropout(0.2))
    model.add(Dense(7, activation=tf.nn.softmax)) #The final number of labels we have from 0 to 6

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) #Compiling the models with adam optimiser
    model.fit(x=x_train, y=imtrainlabeldb, epochs=50) #Epochs number of times data fed
    #Here as NO VALIDATION data was provided I can't know when overfitting occurs.Using a moderate
    # value of EPOCH for training the accuracy were as--- epoch=10(.95),15(0.977),18(0.9835)
    # ,30(0.9916),50(0.9965) Hence taking EPOCH 50 as value. Training error is low, testing may be
    # high but don't have validation data


    n=0
    outputs=[]
    while(n<len(x_test)):
     pred = model.predict(np.asarray(x_test[n]).reshape(1,28,28,1)) #This takes 4d array hence shaping it accordingly

     outputs.insert(n,pred.argmax()) #Here we get the predicted value
     n=n+1

    print(outputs,len(outputs))

    makecsvfile(outputs) #Creating csv file
    # for i in imtrdb:
    #      j=np.array(i).reshape(28,28)
    #      p.append(j.tolist())





def makecsvfile(outputs):

    index=[]

    n=0
    while(n<len(outputs)):
        p=[n,outputs[n]]
        index.insert(n,p)
        n=n+1
    print(index)
    header=['image_index','class']
    with open('outputs.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(i for i in header) #Headers
        n=0


        writer.writerows(index)#Writing in the rows
        #    n=n+1


def loadData():
    # for reading also binary mode is important
    trainImagedb = open('train_image.pkl', 'rb')
    trainLabeldb =open('train_label.pkl','rb')

    testImagedb=open('test_image.pkl','rb')
    imtdb = pickle.load(trainImagedb)
    imldb=pickle.load(trainLabeldb)

    imtsdb=pickle.load(testImagedb)

    mediansitetraindata(imtdb,imldb,imtsdb)


    # n=0
    # while(n<5):
    #
    #    p=np.asarray(imtsdb[n])  #Getting to see the image using PIL
    #    im=Image.fromarray(p.reshape(28,28),'L')
    #    im.show()
    #    n=n+1
    #
    # p = np.asarray(imtsdb[34])  # Getting to see the image using PIL
    # im = Image.fromarray(p.reshape(28, 28), 'L')
    #im.show()
        #print(imdata)
       # print('*************')
        #print(imldata)
       # print('##############')
      #  print(n)
       # n=n+1
       # print('/////////////')

    # for keys in image:
    #
    #    i=keys
    #    print(len(i))







    #print(dbfile)




if __name__ == '__main__':
   loadData()  #loading the data


