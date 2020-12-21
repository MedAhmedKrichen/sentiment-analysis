import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from  tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('/content/drive/MyDrive/IMDB Dataset.csv')as file: 
    
  csvFile = csv.reader(file) 
  next(csvFile)
  npFile=list(csvFile)
  split=int(len(npFile)*0.8)
  train_datset=npFile[:split]
  test_datset=npFile[split:]
  senteces_TRAIN=[]
  label_TRAIN=[]
  for row in train_datset:
    senteces_TRAIN.append(row[0])
    if row[1]=='positive':
      label_TRAIN.append(1)
    else:
      label_TRAIN.append(0)
  senteces_TEST=[]
  label_TEST=[]
  for row in test_datset:
    senteces_TEST.append(row[0])
    if row[1]=='positive':
      label_TEST.append(1)
    else:
      label_TEST.append(0)
  label_TRAIN=np.array(label_TRAIN)
  label_TEST=np.array(label_TEST)


from google.colab import drive
drive.mount('/content/drive')


tokenizer=Tokenizer(num_words=4000,oov_token="OOV")
tokenizer.fit_on_texts(senteces_TRAIN)
sequences=tokenizer.texts_to_sequences(senteces_TRAIN)
word_index=tokenizer.word_index
print(sequences[:4])


test_sequences=tokenizer.texts_to_sequences(senteces_TEST)

padded=pad_sequences(sequences,maxlen=20,padding='post',truncating='pre')
test_padded=pad_sequences(test_sequences,maxlen=20,padding='post',truncating='pre')

model=tf.keras.models.Sequential([
                tf.keras.layers.Embedding(4000,15,input_length=20),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(15)),
                tf.keras.layers.Dense(32,activation='relu'),
                tf.keras.layers.Dense(1,activation='sigmoid')                  

])
callback=tf.keras.callbacks.EarlyStopping('val_loss',patience=10)
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
history=model.fit(padded,label_TRAIN,epochs=100,validation_data=(test_padded,label_TEST),callbacks=[callback])


lt.figure( figsize=(6,4))
plt.plot(range(1,13),history.history['accuracy'] ,label='accuracy')
plt.plot(range(1,13),history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()




plt.figure( figsize=(6,4))
plt.plot(range(1,13),history.history['loss'] ,label='loss')
plt.plot(range(1,13),history.history['val_loss'],label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

