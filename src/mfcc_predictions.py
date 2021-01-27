import random 
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras import Sequential

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


#Copying the directory to \content directory for faster execution
!scp -r /content/drive/My\ Drive/Midas\_task/simple\_mfcc\_data .

SEED=145
PATH = '/content'
TRAIN_DIR = os.path.join(PATH,'simple_mfcc_data','train')
VAL_DIR = os.path.join(PATH,'simple_mfcc_data','val')
OPT = SGD(lr=1e-3)



drive.mount('/content/drive')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_categorical_loss')>0.97):
            print("\nReached 97% validation accuracy so cancelling training!")
            self.model.stop_training = True



model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(64, (2,2), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    #tf.keras.layers.BatchNormalization(axis=-1),
    
    tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    #tf.keras.layers.BatchNormalization(axis=-1),
    
    tf.keras.layers.Conv2D(128,(1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    #tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(256,(1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    #tf.keras.layers.BatchNormalization(axis=-1),
    #tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(256,(1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    
    tf.keras.layers.Conv2D(256,(1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

    tf.keras.layers.Conv2D(256,(1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    #tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.3),
    
    #tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.BatchNormalization(axis=-1),
    #tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dense(5, activation='softmax')
])

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=OPT,
              metrics=['categorical_accuracy'])#,top_3_categ_acc,top_5_categ_acc])

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

# Flow training images in batches of 100 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,  # This is the source directory for training images
        target_size=(150,150),  # All images will be resized to 100x100
        batch_size=16,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(150,150),
        batch_size=16,
        class_mode='categorical')

#Mounting unmounting
#!fusermount -u drive
#!google-drive-ocamlfuse drive
###########

model_path = os.path.join(PATH,'simple_mfcc_data','checkpointAutoSaveModel.h5')

#callbacks = myCallback()
earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=0, mode='max', restore_best_weights=True)
#modelcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_categorical_accuracy', mode='max')
reduce_lr_loss_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

callback_lists = [earlyStopping, reduce_lr_loss_on_plateau ]

history = model.fit_generator(
      train_generator,
      steps_per_epoch=400,  # 6367 images = batch_size * steps
      epochs=100,
      validation_data=validation_generator,
      validation_steps=52,  # 830 images = batch_size * steps
      verbose=1,
      callbacks = callback_lists
      )

train_mfcc = pd.read_csv('/content/train_mfcc.csv',error_bad_lines=False)
val_mfcc = pd.read_csv('/content/val_mfcc.csv',error_bad_lines=False)

train_df_mfcc = pd.DataFrame(train_mfcc['mfccs'].values.tolist())
train_df_label = pd.DataFrame(train_mfcc['label'])

val_df_mfcc = pd.DataFrame(val_mfcc['mfccs'].values.tolist())
val_df_label = pd.DataFrame(val_mfcc['label'])

train_data = pd.concat([train_df_mfcc,train_df_label],axis=1)
val_data = pd.concat([val_df_mfcc, val_df_label],axis=1)

import matplotlib.pyplot as plt
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.savefig(os.path.join('/content/drive/My Drive/Midas_task/simple_mfcc_data/AccPlotv1.png',,dpi=200)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
            
            
            
model.save('/content/drive/My Drive/Midas_task/simple_mfcc_data/modelv1.h5')

train_path = os.path.join(PATH, 'mfcc_nceps13', 'train_data.csv')
val_path = os.path.join(PATH, 'mfcc_nceps13', 'val_data.csv')

train_data = pd.read_csv(train_path, low_memory=False, error_bad_lines = False) #had to use low_memory = False due to many features
val_data = pd.read_csv(val_path, low_memory=False,error_bad_lines = False)

#Taking only first 5000 columns

train_labels = train_data.iloc[:,-1]
val_labels = val_data.iloc[:,-1]

train_data = train_data.iloc[:,1:5000]
val_data = val_data.iloc[:,1:5000]

val_data.fillna(0, inplace=True)
train_data.fillna(0, inplace=True)

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

#One Hot encoding the label vector

lenc = LabelEncoder()
lenc.fit(train_labels)
lenc.transform(train_labels)
train_labels = to_categorical(lenc.transform(train_labels))
val_labels = to_categorical(lenc.transform(val_labels))