import cv2

for sim_no in range(1,4001):
    print('Processing: sim', sim_no)
    for step in range(1,202):
        image = cv2.imread('data/Image-{}-{}.jpg'.format(sim_no, step))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (50,50))
        image = image.reshape(50,50)
        cv2.imwrite('processed/{}/{}.jpg'.format(sim_no,step), image)
