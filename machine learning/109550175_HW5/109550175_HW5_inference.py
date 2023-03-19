import csv
import pandas as pd
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
import keras.utils as image
import numpy as np
import tensorflow as tf
import pandas as pandas
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt 
import gc
from keras.applications import *
import string
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
def write_result(names , labels):
    mid_term_marks = {"filename": [name for name in names],
                    "label":[label for label in labels]}
    mid_term_marks_df = pd.DataFrame(mid_term_marks)
    mid_term_marks_df.to_csv('final_prediction.csv', index=False)

def preprocess(path , is_test = False): # 給出某個目錄下的所有jpg檔案的 向量表示法(in same order) 及其對應到的 labels(to categorical)
  taskname = 'task1/'
  if path.endswith('task2'):
    taskname = 'task2/'
  elif path.endswith('task3'):
    taskname = 'task3/'
  correspond_tensors = []
  testcase_names = []
  if is_test == False: 
    pass
  else:
    for filename in os.listdir(path):
      # print(filename)
      if filename.endswith('.jpg') or filename.endswith('.png'):
        str_ = os.path.join(path , filename)
        start = str_.find('task')
        end = str_.find('.png') + 4
        testcase_name = str_[start : end]
        img = None
        if(path.endswith('task3')):
          img = image.load_img(str_ , target_size = (72 , 96))
        else:
          img = image.load_img(str_ , target_size = (100 , 100))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.0
        correspond_tensors.append(img_tensor)
        testcase_names.append(testcase_name)

    return np.array(correspond_tensors)  , np.array(testcase_names)

x_test_task1 , x_names_task1 = preprocess('/home/undergrad/.kaggle/test/task1' , is_test = True)
x_test_task2 , x_names_task2 = preprocess('/home/undergrad/.kaggle/test/task2' , is_test = True)
x_test_task3 , x_names_task3 = preprocess('/home/undergrad/.kaggle/test/task3' , is_test = True)

mapping = {} 
chars = '0123456789'+string.ascii_lowercase
cnt = 0
for ch in chars:
  mapping[cnt] = ch
  cnt += 1

final_model_for_task1 = models.load_model('./model_weight/part1_model.h5')
x_preds_task1 = []
prediction_task1 = final_model_for_task1.predict(x_test_task1)
for pred in prediction_task1:
   for plc , i in enumerate(pred):
    if i == max(pred):
        x_preds_task1.append(mapping[plc])
x_preds_task1 = np.array(x_preds_task1)


final_model_for_task2 = models.load_model('./model_weight/part2_model.h5')
x_preds_task2 = []
prediction_task2 = final_model_for_task2.predict(x_test_task2)
for pred_0 , pred_1 in zip(prediction_task2[0] , prediction_task2[1]):
  ans0 = None
  ans1 = None
  for plc , i in enumerate(pred_0):
    if i == max(pred_0):
        ans0 = str(mapping[plc])
  for plc , i in enumerate(pred_1):
    if i == max(pred_1):
        ans1 = str(mapping[plc])
  x_preds_task2.append(ans0 + ans1)
x_preds_task2 = np.array(x_preds_task2)

final_model_for_task3 = models.load_model('./model_weight/part3_model.h5')
prediction_task3 = final_model_for_task3.predict(x_test_task3)
# print(prediction_task3[4].shape)
x_preds_task3 = []
for pred_0 , pred_1 , pred_2 , pred_3 in zip(prediction_task3[0] , prediction_task3[1] , prediction_task3[2] , prediction_task3[3]):
  ans0 = None
  ans1 = None
  ans2 = None
  ans3 = None
  for plc , i in enumerate(pred_0):
    if i == max(pred_0):
        ans0 = str(mapping[plc])
  for plc , i in enumerate(pred_1):
    if i == max(pred_1):
        ans1 = str(mapping[plc])
  for plc , i in enumerate(pred_2):
    if i == max(pred_2):
        ans2 = str(mapping[plc])
  for plc , i in enumerate(pred_3):
    if i == max(pred_3):
        ans3 = str(mapping[plc])
  x_preds_task3.append(ans0 + ans1 + ans2 + ans3)
x_preds_task3 = np.array(x_preds_task3)

final_names = np.concatenate((x_names_task1, x_names_task2  , x_names_task3), axis=0)
final_preds = np.concatenate((x_preds_task1, x_preds_task2  , x_preds_task3), axis=0)

write_result(final_names , final_preds)