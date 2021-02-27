import cv2 as cv
import numpy as np
import operator


def detectBlob(img):
    imageFixed = img
    imageFixedGray = cv.cvtColor(imageFixed, cv.COLOR_BGR2GRAY)
    imageFixedBlur = cv.medianBlur(imageFixedGray, 11)
    imageFixedThresh = cv.adaptiveThreshold(imageFixedBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    contour, heirarchy = cv.findContours(imageFixedThresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    leftBlobCordinates = []
    rightBlobCordinates = []
    for c in contour:
        x, y, w, h = cv.boundingRect(c)

        if (x < 70 or x > 1550) and (10 < h < 20 and 30 < w < 40):
            # cv.rectangle(imageFixed, (x, y), (x + w, y + h), (0, 255, 0), 6)
            if x < 70:
                # print(x, y, h, w)
                leftBlobCordinates.append([x + int(w / 2), y + int(h / 2)])
            else:
                rightBlobCordinates.append([x + int(w / 2), y + int(h / 2)])
    # cv.imshow('img', cv.resize(imageFixed, (600, 600)))
    # cv.waitKey(0)
    sortedLeftList = sorted(leftBlobCordinates, key=lambda leftBlobCordinates: (leftBlobCordinates[1]))
    sortedRightList = sorted(rightBlobCordinates, key=lambda rightBlobCordinates: (rightBlobCordinates[1]))
    sortedList = sortedLeftList + sortedRightList
    sortedList = np.asarray(sortedList)
    return sortedList


if __name__ == '__main__':
    imageReference = cv.imread('/home/manish/Desktop/DatasetForOCRonOMR/TrainDatasetCambridge/RawData/image4323.jpg', -1)
    imageSource = cv.imread('/home/manish/Desktop/DatasetForOCRonOMR/TestDatasetJhun/AlignedData/image13253.jpg', -1)
    fixedPoints = detectBlob(imageReference)
    movingPoints = detectBlob(imageSource)
    if len(fixedPoints) != len(movingPoints):
        print(f'Alignment Error...Feature points did not match... {len(fixedPoints)}, {len(movingPoints)}')
    else:
        H, mask = cv.findHomography(fixedPoints, movingPoints, cv.RANSAC, 5.0)
        algnedImage = cv.warpPerspective(imageSource, H, (imageReference.shape[1], imageReference.shape[0]))
        cv.imwrite('/home/manish/Desktop/aligna.jpg', algnedImage)
        cv.imshow('jfjfj', cv.resize(algnedImage, (600, 600), interpolation=cv.INTER_AREA))
    cv.waitKey(0)
