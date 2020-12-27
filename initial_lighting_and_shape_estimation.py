#%%
import numpy as np
import cv2


def ilase(img, num_images = 1):
    
    def getM(img):
        """This function stores all images into one matrix. It converts the images
        to grayscale to reduce the dimensionality.

        :param imagesFolder:
        :type imagesFolder:

        :return M:
        :rtype M:
        """
        image = cv2.imread(img)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert image to grayscale


        return print(image), print(image.shape)
   
    M = getM(img)
    
    U, D, Vt = np.linalg.svd(M, full_matrices = True)
    '''
    # print(f'Shape of U:{U.shape}, D:{D.shape}, Vt:{Vt.shape}')
    R = image_shape[0]
    C = image_shape[1]

    DL = np.zeros((M.shape[0], 4))
    np.fill_diagonal(DL, D[:4]) # this creates a diagonal matrix
    DL = np.sqrt(DL)
    DS = np.zeros((4, M.shape[1]))
    np.fill_diagonal(DS, D[:4]) # this creates a diagonal matrix
    DS = np.sqrt(DS)

    L = U.dot(DL) # L for lighting?
    S = DS.dot(Vt) # S for shape?

    M_4 = L.dot(S)

    ReIm4rank = M_4.T.reshape((R, C, num_images))

    # TODO: Add in functionality to display and save the images
    

    return L, S

    '''
# %%
ilase('/Users/angusharrington/Documents/AEK_total_moving_faces/3D/perfect_dubbing_v1/3D_flow/data/g_bush_01.jpg')
# %%
