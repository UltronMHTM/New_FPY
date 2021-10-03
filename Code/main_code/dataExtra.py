from config import *
import cv2
import xml.etree.ElementTree as ET
import os
import pandas as pd
import random
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.utils import to_categorical
def loadData():
    df = pd.read_csv('../../dataset-master/dataset-master/labels.csv')
    df = df.drop(columns=['Unnamed: 0']).dropna()
    class_set = {"NEUTROPHIL":0, "BASOPHIL":1, "EOSINOPHIL":2, "LYMPHOCYTE":3, "MONOCYTE":4}
    image_set = []
    label_set = []
    for num in range(410):
        image_sample = []
        if (os.path.exists(image_path+'/BloodImage_00%03d.jpg'%num)*\
                len((df[df["Image"]==num].values)))==False: #(os.path.exists(tree_path+'/BloodImage_00%03d.xml'%num))
            continue
        if re.search(r",",(df[df["Image"]==num].values)[0][1]):
            continue
        image_set.append(cv2.imread(image_path+'/BloodImage_00%03d.jpg'%num))
        label_set.append([class_set[(df[df["Image"]==num].values)[0][1]]])#df[df["Image"]==num]["Category"])
        # tree = ET.parse(tree_path+'/BloodImage_00%03d.xml'%num)
    # image_set = tf.data.Dataset.from_tensor_slices(np.array(image_set))
    # label_set = tf.data.Dataset.from_tensor_slices(np.array(label_set)).map(lambda z: tf.one_hot(z,10))
    # total_set = tf.data.Dataset.zip((image_set, label_set)).shuffle(100)
    image_set = np.array(image_set)
    # label_set = np.array([to_categorical(label_set)]) #tf.data.Dataset.from_tensor_slices(np.array(label_set)).map(lambda z: tf.one_hot(z,10))
    label_set = np.array(label_set)
    return image_set, label_set

# loadData()