import cv2


# Reading image from path
def read_image(path):
    img = cv2.imread(path)
    return img


# Extracting features using ORB (Oriented Fast and Rotated Brief)
def features(path):
    img = read_image(path)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)

    feat = cv2.drawKeypoints(img, kp, None)
    return feat


if __name__ == '__main__':
    path_of_file = 'telephone.jpeg'  # path to image
    feat = features(path_of_file)
    cv2.imshow('image', feat)
    cv2.waitKey(0)  # Press 'q' to quit
