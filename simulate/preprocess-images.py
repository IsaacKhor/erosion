import cv2

# Process images
for sim_no in range(1,4001):
    print('Processing: sim', sim_no)
    for step in range(1,202):
        image = cv2.imread('data/5holes/{}/{}.jpg'.format(sim_no, step))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (50,50))
        image = image.reshape(50,50)
        cv2.imwrite('data/5holes-processed/{}/{}.jpg'.format(sim_no,step), image)

"""
# Consolidate into 1 file
import numpy as np
images = list()
for sim_no in range(1,4001):
    print('Processing: sim', sim_no)
    simulation = list()
    for step in range(1,202):
        image = cv2.imread('data/5holes/{}/{}.jpg'.format(sim_no, step))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (50,50))
        image = image.reshape(50,50)
        simulation.append(image)
    images.append(simulation)
data = np.asarray(images)
np.save('data.npy', data)
"""

# Assigns a label from step 200 of a simulation. Accepts a single image which
# it assumes is at the last time step of the simulation, and for the top left
# and top right areas it checks if there are more than n pixels below a 
# threshold
# OR if any of the pixels in the hollow area is below said threshold
def get_label_from_img(img, thres=50, n=4):
    topleft = img[2:10,2:10]
    topright = img[2:10,40:48]
    is_going_left = (topleft<thres).sum() > n or (img[0:2,2:4]<thres).sum() > 1
    is_going_right = (topright<thres).sum() > n or (img[0:2,46,48]<thres).sum() > 1

    if is_going_left and is_going_right:
        return [1, 1]
    elif is_going_right:
        return [0, 1]
    elif is_going_left:
        return [1, 0]
    else:
        return [0.5, 0.5]