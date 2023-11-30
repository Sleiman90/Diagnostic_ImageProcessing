import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def main():

    # initialization of constants
    beta = 0.5
    c = 0.08

    # load and prepare image
    image_rgb = cv2.imread('../data/coronaries.jpg')
    image = convert2gray(image_rgb)

    scales = [1.0, 1.5, 2.0, 3.0]
    images_vesselness = []
    for s in scales:

        images_vesselness.append(calculate_vesselness_2d(image, s, beta, c))

    result = compute_scale_maximum(images_vesselness)
    show_four_scales(image, result, images_vesselness, scales)

'''
# calculate the vesselness filter image (Frangi 1998)
def calculate_vesselness_2d(image, scale, beta, c):

    # create empty result image
    vesselness = np.zeros(image.shape)

    # compute the Hessian for each pixel
    H = compute_hessian(image, scale)

    # get the eigenvalues for the Hessians
    eigenvalues = compute_eigenvalues(H)

    print('Computing vesselness...')

    # compute the vesselness measure for each pixel
    # TODO: loop over the pixels to compute the vesselness image
    # Hint: use the function vesselness_measure (implement it first below)

    print('...done.')
    return vesselness
'''
def calculate_vesselness_2d(image, scale, beta, c):
    # Create an empty result image
    vesselness = np.zeros(image.shape)

    # Compute the Hessian for each pixel
    H = compute_hessian(image, scale)

    # Get the eigenvalues for the Hessians
    eigenvalues = compute_eigenvalues(H)

    print('Computing vesselness...')

    # Loop over the pixels to compute the vesselness image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Get the Hessian matrix at the current pixel
            hessian = H[i, j]

            # Compute the eigenvalues of the Hessian matrix
            lambda1=eigenvalues[i,j,0]
            lambda2=eigenvalues[i,j,1]
            # Compute the vesselness measure for the current pixel
            vesselness[i, j] = vesselness_measure(lambda1, lambda2, beta, c)

    print('...done.')
    return vesselness


def compute_hessian(image, sigma):
    
 
    # Gauss filter the input with the given sigma
    image_gauss = gaussian_filter(image, sigma=sigma, mode='constant')

    print('Computing Hessian...')

    # Compute the first-order gradients
    dx, dy = np.gradient(image_gauss)

    # Compute the second-order derivatives
    dxx = np.gradient(dx)[0]
    dxy = np.gradient(dx)[1]
    dyx = np.gradient(dy)[0]
    dyy = np.gradient(dy)[1]
    # Scale normalization
    scale = sigma**2
    dxx *= scale
    dxy *= scale
    dyx *= scale
    dyy *= scale

    # Assemble the Hessian matrix
    H = np.empty((image_gauss.shape[0], image_gauss.shape[1], 2, 2))
    H[..., 0, 0] = dxx
    H[..., 0, 1] = dxy
    H[..., 1, 0] = dxy
    H[..., 1, 1] = dyy


  

    return H
    


# create array for the eigenvalues and compute them
def compute_eigenvalues(hessian):
                evs = np.zeros((hessian.shape[0], hessian.shape[1], 2))
                for i in range(hessian.shape[0]):
                    for j in range(hessian.shape[1]):
                        hessian_ij = hessian[i, j]
                        if hessian_ij.ndim < 2:
                         hessian_ij = np.atleast_2d(hessian_ij)
            
                        eigvals, _ = np.linalg.eig(hessian_ij)
                        evs[i,j]=eigvals
                

               
               
                return evs

    
'''
In the line eigenvalues[..., 1], the ... is a shorthand notation in NumPy to represent all the axes that are not explicitly mentioned.

In the context of eigenvalues, the Hessian matrix is a 2x2 matrix for each pixel location in the image. The eigenvalues variable represents the eigenvalues of the Hessian matrix computed for each pixel.

Since eigenvalues is a multidimensional array, the ... is used to indicate that we want to access all the values along the remaining axes. In this case, the remaining axes correspond to the spatial dimensions of the image (e.g., height and width).

So, eigenvalues[..., 1] selects the second eigenvalue for all pixel locations in the image, while eigenvalues[..., 0] would select the first eigenvalue. This allows us to access the eigenvalues for all pixels efficiently without explicitly specifying the dimensions.

For example, if eigenvalues is a 3D array of shape (height, width, 2), then eigenvalues[..., 1] would be a 2D array of shape (height, width) containing the second eigenvalues for all pixel locations
'''


# calculate the 2-D vesselness measure (see Frangi paper or course slides)
def vesselness_measure(lambda1, lambda2, beta, c):

    lambda1,lambda2 =sort_descending(lambda1,lambda2)  # Sort eigenvalues in descending order

    if np.any(lambda1 > 0):
        v = 0  # Vesselness measure is zero if lambda1 is positive
    else:
        RB = np.abs(lambda2 / lambda1)  # Radialness/Brightness
        S = np.sqrt(lambda1**2 + lambda2**2)  # Structure
        v = np.exp(-(RB**2) / (2 * beta**2)) * (1 - np.exp(-(S**2) / (2 * c**2)))

    return v


# takes a list of vesselness images and returns the pixel-wise maximum as a result
def compute_scale_maximum(image_list):

    result = image_list[0]
    print('Computing maximum...')

    # TODO: compute the image that takes the PIXELWISE maximum from all images in image_list
    for i in range(1, len(image_list)):
        result = np.maximum(result, image_list[i])

    print('...done.')
    return result


# convert to gray scale and normalize for float
# (OpenCV treats color pixels as BGR)
def convert2gray(image_rgb):

    temp = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image_gray = temp.astype(np.float32) / 255.0

    return image_gray


# rearrange pair of values in descending order
def sort_descending(value1, value2):

    if np.abs(value1) < np.abs(value2):
        buf = value2
        value2 = value1
        value1 = buf

    return value1, value2


# special function to show the images from this exercise
def show_four_scales(original, result, image_list, scales):

    plt.figure('vesselness')

    prepare_subplot_image(original, 'original', 1)
    prepare_subplot_image(image_list[0], 'sigma = '+str(scales[0]), 2)
    prepare_subplot_image(image_list[1], 'sigma = '+str(scales[1]), 3)
    prepare_subplot_image(result, 'result', 4)
    prepare_subplot_image(image_list[2], 'sigma = '+str(scales[2]), 5)
    prepare_subplot_image(image_list[3], 'sigma = '+str(scales[3]), 6)

    plt.show()


# helper function
def prepare_subplot_image(image, title='', idx=1):

    if idx > 6:
        return

    plt.gcf()
    plt.subplot(2, 3, idx)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='gray', vmin=0, vmax=np.max(image))


# function for displaying an image and waiting for user input
def show_image(i, t, destroy_windows=True):

    cv2.imshow(t, i)

    print('Press a key to continue...')
    cv2.waitKey(0)

    if destroy_windows:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
