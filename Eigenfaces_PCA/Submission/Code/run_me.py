import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import sys
from sklearn.cluster import KMeans

def visualize(im1, im2, k):
	# displays two images
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(im1)
    plt.axis('off')
    plt.title('Original')
    f.add_subplot(1,2, 2)
    plt.imshow(im2)
    plt.axis('off')
    plt.title('Cluster: '+str(k))
    plt.savefig('k_means_'+str(k)+'.jpg')
    plt.show()
    return None

# RMSE for KMeans Error (used for KMeans)
def RMSE(Im1, Im2):
	# computes error
	Diff_Im = Im2-Im1
	Diff_Im = np.power(Diff_Im, 2)
	Diff_Im = np.sum(Diff_Im, axis=2)
	Diff_Im = np.sqrt(Diff_Im)
	sum_diff = np.sum(np.sum(Diff_Im))
	avg_error = sum_diff / float(Im1.shape[0]*Im2.shape[1])

    # Mean pixel error between 2 images:
	return avg_error

# Mean Squared error between 2 images: |x - x(n)|^2 (used for PCA)
def MSE(Im1, Im2):
    # Note that Im1 and Im2 are both vectors

    # Note this is mean pixel error between 2 images.
    return np.linalg.norm(Im1 - Im2, 2)**2 / np.shape(Im1)[0]


# grab the image
original_image = np.array(Image.open('../../Data/shopping-street.jpg'))


##########################################################################################################

################################# PCA #####################################

# Return the mean of data:
def mean(X):
    return np.sum(X, axis = 0) / np.shape(X)[0]

# Read the images and store them as row vector in X (100, 2500):
def face_data():
    first_image = np.array(Image.open('../../Data/Faces/face_0.png'), dtype= float)
    flatten_first_image = first_image.reshape(2500)

    data = flatten_first_image

    for i in range(1, 100):
        # Read and flatten the image at i (starting from 1):
        image_i = np.array(Image.open('../../Data/Faces/face_%s.png' % (i)), dtype= float)
        flatten_image_i = image_i.reshape(2500)

        data = np.vstack((data, flatten_image_i))

    print("Shape of data: ", np.shape(data))

    mean_data = mean(data)
    normalised_data = data - mean_data

    return normalised_data

# Returns the D eigenvectors from corresponding to largest to smallest eigenvalues of Covariance matrix:
def eigenvectors_components(X):

    # Calculat the Covariance matrix C:
    num_images = np.shape(X)[0]

    C = np.dot(X.transpose(), X) / num_images

    # Calculate eigenvectors & eigenvalues of C
    evalues, evectors = np.linalg.eigh(C)

    # Sort the eigenvalues (in modulus)
    sort_evalues = np.flip(np.sort(np.abs(evalues)))

    '''
    - Next, we need to store the index that results after sorting
    - Note that np.argmax returns the sorting index in increasing order.
    - Therefore, we must flip it using np.flip
    '''
    sort_index_increasing = np.argsort(np.abs(evalues))
    sort_index_decreasing = np.flip(sort_index_increasing)

    # Initial result matrix of eigenvectors:
    index_first_eigenvalue = sort_index_decreasing[0]

    # Calculate the dimension of an arbitrary eigenvector:
    dim_eigenvector = np.shape(evectors[:, 0])[0]

    largest_eigenvectors = evectors[:, index_first_eigenvalue].reshape(dim_eigenvector, 1)

    for i in range(1, len(sort_index_decreasing)):
        # index in the ith largest eigenvalues:
        index = sort_index_decreasing[i]

        # ith largest eigenvector:
        eigenvector_i = evectors[:, index].reshape(dim_eigenvector, 1)

        # Append the vector to the result:
        largest_eigenvectors = np.hstack((largest_eigenvectors, eigenvector_i))

        if (i % 500 == 0):
            print(i)


    return largest_eigenvectors, sort_evalues, sort_index_decreasing

# PCA:
def PCA(X, k, target):
    W = eigenvectors_components(X)[0]

    # Compressed Representation of X: XW
    # X = [x_1, ..., x_n]^T and W = [w_1, ..., w_k]

    W_k = W[:, range(0, k)]
    compressed_x = np.dot(W_k.transpose(), target)

    # Reconstruction of X: W * (XW)'
    reconstructed_x = np.dot(W_k, compressed_x)

    # Compute Compression Rate:
    compression_rate = (compressed_x.nbytes + W_k.nbytes) / X.nbytes

    # Return image as vector (flatten image)
    return reconstructed_x, compression_rate



########################################## Testing #################################################
# faces_data = face_data()

# K_parameters = [3, 5, 10, 30, 50, 100, 150, 300]


# test_image = np.array(Image.open('../../face.png'))
# flatten_test_image = test_image.reshape(2500)


########## Projecting new image into subspaces of k eigenvectors ###########

# # Projecting new image:
# PCA_errors = []
# f, ax = plt.subplots(3,3)
# ax = ax.ravel()
# ax[0].imshow(test_image, cmap= "gray")
# ax[0].set_title("Original Image", fontsize= 10)
# ax[0].axis('off')


# #Plotting:

# for i, k in enumerate(K_parameters):
#     # Reconstruct the image:
#     reconstructed_image = PCA(faces_data, k, flatten_test_image)[0] + mean(faces_data)
#     reconstructed_image = reconstructed_image.reshape(50,50)

#     # Plot
#     ax[i + 1].imshow(reconstructed_image, cmap="gray")
#     ax[i + 1].set_title("K =  " + str(k), fontsize= 10)
#     ax[i + 1].axis('off')

#     # Error (MSE)
#     PCA_errors.append(RMSE(test_image.reshape(50,50,1), reconstructed_image.reshape(50,50,1)))

# print(PCA_errors)

# plt.tight_layout()
# plt.show()

########### Calculating for MSE for the whole dataset X for different k ############
# PCA_errors_dataset  = []
# compression_rate_dataset = []

# num_images = np.shape(faces_data)[0]
# mean_faces_data = mean(faces_data)


# for k in K_parameters:
#     print("Parameter: ", k)

#     # Calculate projection of entire X:
#     X_projection, compression_rate_k = PCA(faces_data, k, faces_data.transpose()) 
#     X_projection = X_projection.transpose() + mean_faces_data

#     # Compression rate at k:
#     compression_rate_dataset.append(compression_rate_k)
#     print(compression_rate_k)

#     # MSE
#     MSE_k = 0

#     for i in range(num_images):
#         original_image = faces_data[i, :]
#         reconstructed_image = X_projection[i, :]

#         MSE_k += MSE(original_image, reconstructed_image)

#     # Calculate the average error
#     MSE_k = MSE_k / num_images

#     PCA_errors_dataset.append(MSE_k)

# plt.plot(K_parameters, PCA_errors_dataset)
# plt.xlabel("Value of K")
# plt.ylabel("MSE of X and " r'$\hat{X}$')
# plt.title("MSE of X and " r'$\hat{X}$' " w.r.t k")
# plt.show()



########################################## K-MEANS ###############################################

# # Flatten the image 3D matrix into 2D matrix of all the points
# shopping_image = np.array(Image.open('../../Data/shopping-street.jpg'))

# flatten_shopping_image = shopping_image.ravel().reshape(shopping_image.shape[0] * shopping_image.shape[1], shopping_image.shape[2])
# print(np.shape(flatten_shopping_image))

# Compression using K-Means:
def k_means_compression(k, flatten_image):
    kmeans = KMeans(n_clusters= k)
    kmeans = kmeans.fit(flatten_image)

    prediction = kmeans.predict(flatten_image)

    reconstructed_image = np.zeros(flatten_image.shape)

    for pixel in range(flatten_image.shape[0]):
        # Get the centroid for the correponding pixel:
        centroid = kmeans.cluster_centers_[prediction[pixel]]

        # Assign the original pixel to the corresponding centroid:
        reconstructed_image[pixel] = centroid

    # Calculate the error:
    error = MSE(reconstructed_image, flatten_image)
    #error = RMSE(reconstructed_image.reshape(309, 393, 3), flatten_image.reshape(309, 393, 3))

    return reconstructed_image.reshape(309, 393, 3), error, reconstructed_image


# ############################## Testing ############################
# K_parameters = [2, 5, 10, 25, 50, 100, 200, 15]
# results = []
# kmeans_errors = []
# compression_rate = []

# # Plotting
# f, ax = plt.subplots(3, 3)
# ax[0, 0].imshow(shopping_image)
# ax[0, 0].axis('off')
# ax[0, 0].set_title("Original Image",  fontsize=10)


# for i, k in enumerate(K_parameters):
#     print("Parameter: ", k)
    
#     reconstructed_image, error, flattened_reconstructed_image = k_means_compression(k, flatten_shopping_image)

#     # For plotting using imshow(), we need to convert this into float [0-1] by data normalization
#     reconstructed_image = reconstructed_image / 255

#     results.append(reconstructed_image)
#     kmeans_errors.append(error)

#     # Compression rate:
#     pixels = np.shape(flatten_shopping_image)[0]
#     compression_rate_k = (k * 3 * 32 + pixels * np.log2(k)) / (pixels * 24)
#     print("Compression Rate: ", compression_rate_k)

#     i = i + 1

#     ax[int((i - i % 3) / 3), int(i - int(i / 3) * 3)].imshow(reconstructed_image)
#     ax[int((i - i % 3) / 3), int(i - int(i / 3) * 3)].set_title("K = " + str(k), fontsize=10)
#     ax[int((i - i % 3) / 3), int(i - int(i / 3) * 3)].axis('off')

# plt.tight_layout()
# print()
# print("---------------- K-Means Analysis ------------------")
# print("Error: ", kmeans_errors)
# plt.show()


    








