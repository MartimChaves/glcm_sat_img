from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

verbose = True

# load an image
# mount everest: https://artsandculture.google.com/story/kAWhL0aq-oU_-Q
# sea: https://www.istockphoto.com/pt/foto/blue-sea-water-background-atlantic-gm1136870172-302935743
if verbose: print("Loading images...")
sky = plt.imread("pexels-kasia-palitava-10812242.jpg")
umbrella = plt.imread("pexels-parviz-besharat-pur-8566320.jpg") #александр
if verbose: print("Images loaded")

# get img grey lvl
if verbose: print("Converting RGB imgs to grey lvl (8bit)")
sky_g = np.average(sky,axis=2).astype(np.uint8)
umbrella_g = np.average(umbrella,axis=2).astype(np.uint8)
if verbose: print("Images converted")

log_img = True
finl_img = umbrella_g
img_name = 'umbrella'
glcm_name = 'umbrella_glcm'

plt.imshow(finl_img, cmap='gray')
plt.axis('off')
plt.savefig(f'{img_name}.png',bbox_inches='tight',pad_inches = 0)
plt.clf()

# choose a positional operator
pos_op = [1,0]

# create glcm array
glcm = np.zeros([256,256])

# iterate over image and complete glcm
if verbose: print("Calculating GLCM")
for i in tqdm(range(finl_img.shape[0])): # row
    for j in range(finl_img.shape[1]): # col
        init_val = finl_img[i,j]
        try:
            target = finl_img[i+pos_op[0],j+pos_op[1]]
        except IndexError:
            continue # out of img bounds
        glcm[init_val,target]+=1
        
glcm = glcm/np.sum(glcm)


if verbose: print("Plotting image")
if log_img:
    imgplot = plt.imshow(np.log(glcm+1e-6), cmap='gray')
    plt.axis('off')
    plt.savefig(f'{glcm_name}.png',bbox_inches='tight',pad_inches = 0)
    plt.clf()
else:
    imgplot = plt.imshow(glcm, cmap='gray')
    plt.axis('off')
    plt.savefig(f'{glcm_name}.png',bbox_inches='tight',pad_inches = 0)
    plt.clf()
    
is_symmetric = False
if is_symmetric:
    test_glcm = np.copy(glcm)
    for i in range(test_glcm.shape[0]):
        for j in range(test_glcm.shape[1]):
            if i < j:
                continue
            else:
                temp_val = test_glcm[i,j].copy()
                test_glcm[i,j]-= test_glcm[j,i]
                test_glcm[j,i] -= temp_val

    if verbose: print("Is the GLCM symmetric?")
    imgplot = plt.imshow(test_glcm)
    plt.show()
    print("Number of different values in diagonally subtracted glcm:",len(np.unique(test_glcm,return_counts=True)[1]))
