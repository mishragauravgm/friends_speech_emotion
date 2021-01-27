import pandas as pd
import numpy as np
import random

import librosa as lbr
from librosa.display import waveplot, specshow
from librosa.feature import mfcc

import imageio
import matplotlib.image as mpimg 

import os
import matplotlib.pyplot as plt
%matplotlib inline

PATH = '/content/drive/My Drive/Midas_task'
CLASSES = ['disgust','fear','happy','neutral','sad']

#directories = ['train','val']
drive.mount('/content/drive') #put this because there was some mounting error(file not exist)

new_fold_name = 'mfcc_nceps13' #earlier simple_mfcc_data

train_mfcc = pd.DataFrame(columns = ['mfccs','label']);
val_mfcc = pd.DataFrame(columns = ['mfccs','label']);

location = 0; #To keep track of index

directories = ['train','val']

try: 
    os.mkdir(os.path.join(PATH,new_fold_name)) 
except OSError as error: 
    print('Printing Error',error)   
    
img_dir = os.path.join(PATH,new_fold_name);

for directory in directories:
  
  location = 0;
  
  print(f'Processing {directory}')
  
  try:
    os.mkdir(os.path.join(img_dir,directory))
  except OSError as error:
    print('Printing Error',error)
    
  for label in CLASSES:
    
    try:
      os.mkdir(os.path.join(img_dir,directory,label))
    except OSError as error:
      print('Printing Error',error);
      
    print(f'Processing {directory}/{label}');
    file_path = os.path.join(PATH,'meld',directory,label);
    #print('*******',file_path)
    for files in os.listdir(file_path):
      #print(files)
      filename = os.path.join(file_path,files);
      data,sr = lbr.load(filename);
      mfccs = mfcc(y=data,sr=sr, n_mfcc = 13);
      
      file_without_ext = os.path.splitext(os.path.basename(filename))[0]
      destination = os.path.join(img_dir,directory,label,file_without_ext+'.jpg')
      
      mfcc_reshaped = np.reshape(mfccs.T,(mfccs.shape[0]*mfccs.shape[1],1)).T
      
      print(mfcc_reshaped.shape)
      if(directory =='train'):
        train_mfcc.loc[location,'mfccs'] = mfcc_reshaped[0];
        train_mfcc.loc[location,'label'] = label;
        location=location+1;
      elif(directory =='val'):
        val_mfcc.loc[location,'mfccs'] = mfcc_reshaped[0];
        val_mfcc.loc[location,'label'] = label;
        location=location+1;
      #imageio.imwrite(destination,mfccs)

      

train_mfcc.to_csv('/content/drive/My Drive/Midas_task/mfcc_nceps13/train_mfcc.csv',index=False)

val_mfcc.to_csv('/content/drive/My Drive/Midas_task/mfcc_nceps13/val_mfcc.csv',index=False)

train_mfcc.to_csv('train_mfcc.csv',index=False)

val_mfcc.to_csv('val_mfcc.csv',index=False)


train_df_mfcc = pd.DataFrame(train_mfcc['mfccs'].values.tolist())
train_df_label = pd.DataFrame(train_mfcc['label'])
val_df_mfcc = pd.DataFrame(val_mfcc['mfccs'].values.tolist())
val_df_label = pd.DataFrame(val_mfcc['label'])
train_data = pd.concat([train_df_mfcc,train_df_label],axis=1)
val_data = pd.concat([val_df_mfcc, val_df_label],axis=1)
train_data = pd.concat([train_df_mfcc,train_df_label],axis=1)
val_data = pd.concat([val_df_mfcc, val_df_label],axis=1)

