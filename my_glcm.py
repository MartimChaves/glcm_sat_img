from matplotlib import pyplot as plt
import numpy as np

verbose = True

# load an image
# mount everest: https://artsandculture.google.com/story/kAWhL0aq-oU_-Q
# sea: https://www.istockphoto.com/pt/foto/blue-sea-water-background-atlantic-gm1136870172-302935743
if verbose: print("Loading images...")
mnt_ev = plt.imread("imgs/mounteverest.jpg")
sea = plt.imread("imgs/sea.jpg")
if verbose: print("Images loaded")

# get img grey lvl
if verbose: print("Converting RGB imgs to grey lvl (8bit)")
mnt_ev_g = np.average(mnt_ev,axis=2).astype(np.uint8)
sea_g = np.average(sea,axis=2).astype(np.uint8)
if verbose: print("Images converted")

finl_img = mnt_ev_g

# choose a positional operator
pos_op = [1,0]

# create glcm array
glcm = np.zeros([256,256])

# iterate over image and complete glcm
if verbose: print("Calculating GLCM")
for i in range(finl_img.shape[0]): # row
    for j in range(finl_img.shape[1]): # col
        init_val = finl_img[i,j]
        try:
            target = finl_img[i+pos_op[0],j+pos_op[1]]
        except:
            continue # out of img bounds
        glcm[init_val,target]+=1
        
glcm = glcm/np.sum(glcm)

if verbose: print("Plotting image")
imgplot = plt.imshow(np.log(glcm+1e-6))
plt.show()



