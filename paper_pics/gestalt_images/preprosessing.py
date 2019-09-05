# this approach failed, pass



import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

image_filenames = ['brush2.png', 'fish_fin2.png', 'rocket2.png', 'spin2.png', 'yoyo2.png'] 
images = []
for name in image_filenames: 
    images.append(cv2.imread(name)) 

masks = [] 

for i in range(5): 
    h, w, c = images[i].shape
    mask = 255*np.ones(images[i].shape, dtype=np.uint8) 
    for j in range(h): 
        for k in range(w): 
            if np.all(images[i][j,k,:] == [255,0,255]):
                mask[j,k,:] = [0,0,0] 
    masks.append(mask)
    mask_name = image_filenames[i].split('.')[0]+'_mask' + ".png"
    cv2.imwrite(mask_name, mask)


# plt.imshow(images[0]) 
# plt.show()











