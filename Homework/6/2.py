import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time



# change different K values
n_colors = 64
#n_colors = 256


# use random selection, 1 trial only
init_method = 'k-means++'       #default
init_trials = 10                #default
#init_method = 'random'
#init_trials = 1

# change random_state
randomstate = None                 #default 

# different value of "w"
weight = 0.25
#weight = 0.5
#weight= 1





# Load the Summer Palace photo
china = load_sample_image("china.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(china, dtype=np.float64) / 255

#%%
# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))
width = float(w)
height = float(h)
imarray_with_coordinate = list(np.zeros((w*h, 5)))
ii = np.zeros((w,1))
jj = np.zeros((h,1))
for i in range(w):
    ii[i] = (i/width) *255 * weight
    for j in range(h):    
        jj[j] = (j/height) *255 * weight
        imarray_with_coordinate[i*j][:3] = list(image_array[i*j])
        imarray_with_coordinate[i*j][3] = ii[i]
        imarray_with_coordinate[i*j][4] = jj[j]
#        image_array[i*j] = np.asarray(imarray_with_coordinate[i*j])
#        if imarray_with_coordinate[i*j] == np.zeros((5,1)):
#            print('error in %0.3f, %0.3f' %i %j)
image_array = np.asarray(image_array)
        

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(imarray_with_coordinate, random_state=randomstate)[:1000]
kmeans = KMeans(n_clusters=n_colors, init=init_method, n_init=init_trials,
                verbose=1, random_state=randomstate).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(imarray_with_coordinate)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(imarray_with_coordinate, random_state=randomstate)[:n_colors + 1]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          imarray_with_coordinate,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = 3
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]][:3]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (%0.3f colors, K-Means)' %n_colors)
plt.imshow(abs(recreate_image(kmeans.cluster_centers_, labels, w, h)))

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (%0.3f colors, Random)' %n_colors)
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()

#plt.figure(4)
#plt.title('error reduction curve of K-Means')
#plt.plot(error)
#plt.imshow
