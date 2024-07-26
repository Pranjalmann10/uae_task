import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import spectral
from bm4d import bm4d

#loading of the dataset
input_dir='Indian_pines.mat'
content = sio.loadmat(input_dir)
print(content.keys())
key='indian_pines'
hcube = content[key]

#loading of the label data
input_dir = 'Indian_pines_gt.mat'
content = sio.loadmat(input_dir)
print(content.keys())
key = 'indian_pines_gt'
gt_label = content[key]

#normalising of the data
normalized_hcube = hcube / np.max(hcube)

#applying the bm4d denosing 
denoised_hcube = bm4d(normalized_hcube, sigma_psd=0.3)

#rescaling of the denoised data back to orignal range
denoised_hcube = denoised_hcube * np.max(hcube)


#select training and test data
gt_locs = np.where(gt_vector !=0)[0]
class_labels = gt_vector[gt_locs]
train_locs,test_locs = train_test_split(gt_locs, test_size=0.2)


#train SVM MODEL
model = SVC(decision_function_shape='ovo')
model.fit(data_vector[train_locs], gt_vector[train_locs])
model_label = model.predict(data_vector[test_locs])
svm_accuracy = accuracy_score(gt_vector[test_locs], model_label)
print(f"overall accuracy (OA) of the test  data using SVM={svm_accuracy}")
#final overall accuacry on the indian pines dataset was achived to be approx 60%
#despite fine tunning many parameters like sigma psd and test size etc overall was not able to increase the accuracy of the model
