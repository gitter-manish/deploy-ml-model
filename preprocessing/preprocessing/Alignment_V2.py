import cv2 as cv
import os
from ImageAlignment import BlobDetection
import time

start_time = time.time()
imageFixed = cv.imread('/home/manish/Desktop/DatasetForOCRonOMR/TrainDatasetCambridge/RawData/image4323.jpg', -1)
fixedPoints = BlobDetection.detectBlob(imageFixed)
count = 0
faultyOmr = []
for entry in os.scandir('/home/manish/Desktop/DatasetForOCRonOMR/TestDatasetJhun/AlignedData'):
    if entry.path.endswith('.jpg') and entry.is_file():
        path = entry.path
        docId = path[path.find('image')+5:path.find('.jpg')]
        # print(docId)
        imageMoving = cv.imread(path, -1)
        movingPoints = BlobDetection.detectBlob(imageMoving)
        if len(movingPoints) == len(fixedPoints):
            H, mask = cv.findHomography(movingPoints, fixedPoints, cv.RANSAC, 5.0)
            alignedImage = cv.warpPerspective(imageMoving, H, (imageFixed.shape[1], imageFixed.shape[0]))
            cv.imwrite('/home/manish/Desktop/JhunAlignedWithBlob/image'+str(docId)+'.jpg', alignedImage)
            count += 1
        else:
            cv.imwrite('/home/manish/Desktop/JhunAlignedWithBlob/image' + str(docId) + '.jpg', alignedImage)
            faultyOmr.append(str(docId))
        print(f'Done processing: {count} file. Last image was: {docId}')
# print(f'There was {len(faultyOmr)} faulty omr sheets for alignment.')
# print(faultyOmr)
print(f'Total time taken is: {(time.time()-start_time)/60} minutes.')

