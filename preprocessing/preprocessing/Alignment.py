import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

img1 = cv.imread('/home/manish/Desktop/image4323.jpg', -1)  # referenceImage

for entry in os.scandir('/home/manish/Desktop/DatasetForOCRonOMR/TestDatasetJhun/RawData'):
    if entry.path.endswith(".jpg") and entry.is_file():
        src = str(entry.path)
        startn = src.find('image')
        endn = src.find('.jpg')
        docid = src[startn:endn]
        dest = '/home/manish/Desktop/DatasetForOCRonOMR/TestDatasetJhun/RawData/image13127.jpg' + src[startn:]

        img2 = cv.imread('/home/manish/Desktop/DatasetForOCRonOMR/TestDatasetJhun/RawData/image13127.jpg', -1)  # sensedImage

        # Initiate AKAZE detector
        akaze = cv.AKAZE_create()

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = akaze.detectAndCompute(img1, None)
        kp2, des2 = akaze.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
        sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

        # Compute homography
        H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC, 5.0)

        # Warp image
        warped_image = cv.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
        cv.imwrite('/home/manish/Desktop/image13127.jpg', warped_image)
        del warped_image
        del img2
        print(f'Done for {src[startn:endn]}')
        break
    break
