import tensorflow as tf
import scipy.io as sio
import segmentation-models as sm
import glob
import cv2 
import os
import numpy as np
from matplotlib import pyplot as plt

BACKBONE= 'resnet34'
preprocess_input=sm.get_preprocessing(BACKBONE)


#RESSIZING IMAGES WHICH IS OPTIONAL, CNN ARE OK WITH LARGE IMAGES
SIZE_X=145
SIZE_Y=145

#CAPTURE TRAINING IMAGES INFO AS A LIST
train_images=[]

# Loop through each .mat file in the directory
for mat_path in glob.glob(os.path.join(directory_path, "*.mat")):
    mat_data = sio.loadmat(mat_path)
    
    if 'IMAGES' in mat_data:
        img = mat_data['IMAGES']
        train_images.append(img)

#convert list to array for machine learning processing
train_images = np.array(train_images)

directory_path='location of the mask images'
#capture masks/label info as  list
train_masks = []
for math_data in glob.glob(os.path.join(directory_path, "*.mat")):
    mat_data = sio.loadmat(mat_path)
    if 'MASKS' in mat_data:
        img = mat_data['MASKS']
        train_masks.append(img)
        
train_masks=np.array(train_masks)


#use x-train and y_train varibale
X=train_images
Y=train_masks
y=np.expand_dims(Y,axis=3)

from sklearn.model_selection import train_test_split
x_train, x_val , y_train,y_val = train_test_split(X,Y, test_size=0.2, random_state=42)

#preprocess input
x_train = preprocess_input(x_train)
x_val= preprocess_input(x_val)

#define model
model=sm.Unet(BACKBONE, encoder_weights='imagnet')
model.compile(optimizer='adam', loss='sm.losses.bce_jaccard_loss', metrics=['sm.metrics.iou_score'])

print(model.summary())

history=model.fit(x_train,y_train,batch_size=8,epochs=10,verbose=1,validation_data=(x_val,y_val))
accuracy = model.evaluate(x_val,y_val)
print(accuracy)


#plot the training and validation accuracy and loss at each epochs
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs, loss , 'y' , label='training loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
