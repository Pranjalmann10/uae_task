import numpy as np
import scipy.io as sio
from skimage import io, exposure
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import spectral


#loading of tha data
input_dir = '/content/Indian_pines .mat'
content = sio.loadmat(input_dir)
#print(content.keys())
key = 'indian_pines'
hcube = content[key] 

#loading of the label
input_dir = '/content/Indian_pines_gt.mat'
content = sio.loadmat(input_dir)
print(content.keys())
key = 'indian_pines_gt'
gt_label = content[key]


#removing of the spicific bands to reduce the noise from the hyposcpectral data 
#we came to know about the removing of specific bands suchs doing tests like  snr ratio,visula inspection and correalation analysis
num_classes = 16 
band_indices = list(range(104,109)) + list(range(150,164)) 
new_hcube = np.delete(hcube, band_indices, axis=2)

rgb_img = spectral.get_rgb(new_hcube, [29,19,9])
#this line of code helps  us to generate an rgb image of the hposctral data using the specific bandss(29,19,9) for the red,green and blue channels


# thes linese of code apply the gaussian filtering to each band of the hypospectral dataset to reduce noise
from scipy.ndimage import gaussian_filter
hs_data = new_hcube
M, N, C = hs_data.shape
hs_data_filtered = np.zeros_like(hs_data, dtype=np.uint8)

for band in range(C):
    band_image = hs_data[:,:,band]
    band_image_filtered = gaussian_filter(band_image, sigma=2)
    band_image_gray = exposure.rescale_intensity(band_image_filtered, out_range=(0,255))
    hs_data_filtered[:,:,band] = band_image_gray.astype(np.uint8)


#reshape data for the classification
data_vector = hs_data_filtered.reshape(-1,C)
gt_vector = gt_label.flatten()

#training of the dataset
gt_locs = np.where(gt_vector != 0 )[0]
class_labels = gt_vector[gt_locs]

train_locs, test_locs = train_test_split(gt_locs, test_size=0.3, random_state=42)

#print(data_vector.shape)  
#print(class_labels.shape)


#svm model
svm_model = SVC(decision_function_shape='ovo')
svm_model.fit(data_vector[train_locs], gt_vector[train_locs])

svm_label_out = svm_model.predict(data_vector[test_locs])
svm_accuracy = accuracy_score(gt_vector[test_locs], svm_label_out)
print(f"Overall Accuracy(OA) of the test data using the Svm= {svm_accuracy} ")
# the overall accuracy of the indian_pine_dataset was found to be 0.8865040650406504 

                                         
