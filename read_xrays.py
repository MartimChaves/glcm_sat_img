import tensorflow as tf
import numpy as np
import time
from matplotlib import pyplot as plt
import os
import cv2
import pandas as pd
from tqdm import tqdm

def read_imgs_class(dataset, indx):

    labels = np.zeros((112120,15))

    for idx, element in enumerate(dataset.as_numpy_iterator()): #112120 total imgs :o
        labels[idx*bs:(idx+1)*bs] = element[1]

    class_lbl = np.zeros(15)
    class_lbl[indx] = 1
    class_ind = np.where(labels == class_lbl)[0]


    # save labels
    # use labels to read specific imgs
    t = time.process_time()
    arr = class_ind #np.array([3, 4, 5])
    my_table = {x:True for x in class_ind} #{3: True, 4: True, 5: True}

    dataset = dataset.enumerate()
    keys_tensor = tf.constant(arr)
    vals_tensor = tf.ones_like(keys_tensor)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=0)  # If index not in table, return 0.

    def hash_table_filter(index, value):
        table_value = table.lookup(index)  # 1 if index in arr, else 0.
        index_in_arr =  tf.cast(table_value, tf.bool) # 1 -> True, 0 -> False
        return index_in_arr

    filtered_ds = dataset.filter(hash_table_filter)

    final_ds = filtered_ds.map(lambda idx,value: value)
    elapsed_time = time.process_time() - t

    print(f"Elapsed time: {elapsed_time}")
    
    return final_ds

feature_map = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_id': tf.io.FixedLenFeature([], tf.string),
    'No Finding': tf.io.FixedLenFeature([], tf.int64),
    'Atelectasis': tf.io.FixedLenFeature([], tf.int64),
    'Consolidation': tf.io.FixedLenFeature([], tf.int64),
    'Infiltration': tf.io.FixedLenFeature([], tf.int64),
    'Pneumothorax': tf.io.FixedLenFeature([], tf.int64),
    'Edema': tf.io.FixedLenFeature([], tf.int64),
    'Emphysema': tf.io.FixedLenFeature([], tf.int64),
    'Fibrosis': tf.io.FixedLenFeature([], tf.int64),
    'Effusion': tf.io.FixedLenFeature([], tf.int64),
    'Pneumonia': tf.io.FixedLenFeature([], tf.int64),
    'Pleural_Thickening': tf.io.FixedLenFeature([], tf.int64),
    'Cardiomegaly': tf.io.FixedLenFeature([], tf.int64),
    'Nodule': tf.io.FixedLenFeature([], tf.int64),
    'Mass': tf.io.FixedLenFeature([], tf.int64),
    'Hernia': tf.io.FixedLenFeature([], tf.int64)
}

# 2. define decoding function for image and tfrecord
def image_decoder(data):
    example = tf.io.parse_single_example(data, feature_map) 
    image = example['image']
    image = tf.io.decode_image(image)#,channels=3)
    #image = tf.image.convert_image_dtype(image, tf.float32)
    #image = tf.image.resize_with_pad(image, 500, 500) #150,150 #768, 768
    #image.set_shape([500,500,1])#,3]) #150,150,1
    #image = image/tf.math.reduce_max(image) #np.max(image) #255.
    
    print([label for label in sorted(list(example.keys())) if label!='image' and label!='image_id'])
    labels = [tf.cast(example[x], tf.float32) for x in sorted(list(example.keys())) if x!='image_id' and x!='image']
    
    return image, labels

# 3. create dataset
bs = 128
datadir = "./data/chest_xrays/archive/data/"
data_list = [os.path.join(datadir,x) for x in os.listdir(datadir)]
dataset = tf.data.TFRecordDataset(data_list)
dataset = dataset.map(image_decoder, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(bs).prefetch(tf.data.experimental.AUTOTUNE)


num_classes_dict = {x:0 for x in range(15)} # 15 classes
csv_info = {'image':[],'labels':[]}
num_save_imgs = 1000
save_root = "./data/xrays/train_images/"
img_counter = 0

available_lbls = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
                  'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
                  'Infiltration', 'Mass', 'No Finding', 'Nodule',
                  'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
available_lbls = np.array(available_lbls,dtype=object)

for element in tqdm(dataset.as_numpy_iterator()): #112120 total imgs :o
    
    imgs, lbls = element[0], element[1]

    for img, lbl in zip(imgs,lbls):
        for possible_lbls in np.where(lbl==1)[0]:
            if num_classes_dict[possible_lbls] < num_save_imgs:
                # img path
                img_name = f"{img_counter}.jpg"
                img_counter += 1
                img_path = os.path.join(save_root,img_name)
                
                # save img
                cv2.imwrite(img_path, img)
                
                # save img info
                for lbl_ind in np.where(lbl==1)[0]:
                    num_classes_dict[lbl_ind] += 1
                csv_info['image'].append(img_name)
                img_labls = ' '.join(available_lbls[np.where(lbl==1)[0]])
                if img_labls == "No Finding":
                    img_labls = "No_Finding"
                csv_info['labels'].append(img_labls)
                break # img saved
            else:
                pass

            
    all_clss_flag = True
    
    for lbl, num_lbl in num_classes_dict.items():
        if num_lbl < num_save_imgs:
            all_clss_flag = False
            break
        else:
            continue
        
    if all_clss_flag: break
            
df = pd.DataFrame.from_dict(csv_info)
df.to_csv("./data/xrays/train.csv", index=False)
