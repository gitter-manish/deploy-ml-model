import cv2 as cv
import numpy as np
import pandas as pd
import time
import os

start_time = time.time()
lengi = 0
df = pd.DataFrame(columns=['document_id', 'set_id', 'student_id', 'answer_keys'])
xForQuads = [(344, 552), (620, 828), (897, 1106), (1174, 1382)]
yForQuads = [(897, 1115), (1115, 1333), (1333, 1550), (1550, 1767), (1767, 1984)]
setCords = [480, 840, 856, 944]
studentCords = [480, 840, 970, 1378]

thresholdForWhitePixels = 0.195

def getAnswerKeys(img):
    imageOriginal = img

    allScore = []
    answer_keys = ''
    for vert in xForQuads:
        for hor in yForQuads:
            image = imageOriginal[hor[0]:hor[1], vert[0]:vert[1]]
            # <<<<<<<<<<<<<<<<---------------- Preprocessing and removing channels ---------->>>>>>>>>>
            h, w = image.shape[1], image.shape[0]
            image = image[20:h-10, :]
            rows = np.array_split(image, 5, axis=0) #<<<<<<<<<<<<<----------------- Split into image into 5 rows ------------------------>>>>>>>>>>>
            r, a, b, c = 1, 0.25, 0.5, 1
            localKernel = np.array([[a, a, a, a, a, a],
                                    [a, b, b, b, b, a],
                                    [a, b, c, c, b, a],
                                    [a, b, c, c, b, a],
                                    [a, b, b, b, b, a],
                                    [a, a, a, a, a, a]], dtype='float16')
            for row in rows:
                cols = np.array_split(row, 4, axis=1)  #<<<<<<<<<<<<<<<<-------------------- Splits each row into 4 Cells
                tempList = []
                empty = True
                for col in cols:
                    col = cv.resize(col, (36, 36), interpolation=cv.INTER_LINEAR)
                    hCol, wCol = col.shape[1], col.shape[0]
                    center = [int(hCol / 2), int(wCol / 2)]
                    lowerH, upperH, lowerW, upperW = center[0] - 18, center[0] + 18, center[1] - 18, center[0] + 18
                    col = col[lowerH:upperH, lowerW:upperW]
                    ################################ Internal division of a cell in 6x6 #########################
                    colMatrix = np.zeros((6, 6), dtype='float16')
                    colRowGrids = np.array_split(col, 6, axis=0)   #<<<<<<---------------- Splits each cell into 6 rows
                    i = 0

                    for colRowGrid in colRowGrids:
                        totalNzcValue = 36
                        colGrids = np.array_split(colRowGrid, 6, axis=1) #<<<<<<---------------- Splits each cell row into 6 smaller cells of size 6x6
                        j = 0
                        if totalNzcValue != 0:
                            for colGrid in colGrids:
                                nzc = np.count_nonzero(colGrid)
                                colMatrix[i][j] = float(nzc/totalNzcValue)
                                j += 1
                        i += 1
                    aggVal = np.round(np.sum(colMatrix*localKernel)/np.sum(localKernel), 4)
                    allScore.append(aggVal)
                    tempList.append(aggVal)
                for index in range(len(tempList)):
                    if tempList[index] > thresholdForWhitePixels:
                        empty = False
                        answer_keys += str(index + 1)
                if empty == True:
                    answer_keys += str('x')
                answer_keys += '#'
    return answer_keys

def getIds(image, num):
    allScore = []
    reqString = ''
    r, a, b, c = 1, 0.25, 0.5, 1
    localKernel = np.array([[a, a, a, a, a, a],
                            [a, b, b, b, b, a],
                            [a, b, c, c, b, a],
                            [a, b, c, c, b, a],
                            [a, b, b, b, b, a],
                            [a, a, a, a, a, a]], dtype='float16')

    rows = np.array_split(image, num,
                          axis=1)  # <<<<<<<<<<<<<----------------- Split into image into 5 rows ------------------------>>>>>>>>>>>

    for row in rows:
        row = cv.resize(row, (44, 360), interpolation=cv.INTER_AREA)
        cols = np.array_split(row, 10, axis=0)  # <<<<<<<<<<<<<<<<-------------------- Splits each row into 4 Cells
        # cv.imshow('col', row)

        tempList = []
        empty = True
        for col in cols:
            col = cv.resize(col, (36, 36), interpolation=cv.INTER_LINEAR)
            hCol, wCol = col.shape[1], col.shape[0]
            center = [int(hCol / 2), int(wCol / 2)]
            lowerH, upperH, lowerW, upperW = center[0] - 18, center[0] + 18, center[1] - 18, center[0] + 18
            col = col[lowerH:upperH, lowerW:upperW]
            ################################ Internal division of a cell in 6x6 #########################
            colMatrix = np.zeros((6, 6), dtype='float16')
            colRowGrids = np.array_split(col, 6, axis=0)  # <<<<<<---------------- Splits each cell into 6 rows
            i = 0

            for colRowGrid in colRowGrids:
                totalNzcValue = 36
                colGrids = np.array_split(colRowGrid, 6,
                                          axis=1)  # <<<<<<---------------- Splits each cell row into 6 smaller cells of size 6x6
                j = 0
                if totalNzcValue != 0:
                    for colGrid in colGrids:
                        nzc = np.count_nonzero(colGrid)
                        colMatrix[i][j] = float(nzc / totalNzcValue)
                        j += 1
                i += 1
            aggVal = np.round(np.sum(colMatrix * localKernel) / np.sum(localKernel), 4)
            allScore.append(aggVal)
            tempList.append(aggVal)
        blank = True
        for index in range(len(tempList)):
            if tempList[index] > thresholdForWhitePixels:
                blank = False
                reqString += str(index)
        if blank == True:
            reqString += '$'
    if reqString == '':
        if num == 9:
            reqString = '$$$$$$$$$'
        else:
            reqString = '$$'
    return reqString

def image_thresholding(imageOriginal):
    imageOriginal = imageOriginal.copy()
    imageHSV = cv.cvtColor(imageOriginal, cv.COLOR_BGR2HSV)
    lBound = (0, 0, 135)
    uBound = (255, 255, 255)
    mask = cv.inRange(imageHSV, lBound, uBound)
    imageWithoutChannels = cv.bitwise_and(imageOriginal, imageOriginal, mask=mask)
    imageGray = cv.cvtColor(imageWithoutChannels, cv.COLOR_BGR2GRAY)
    _, imageThresh = cv.threshold(imageGray, 80, 255, cv.THRESH_BINARY_INV)

    return imageThresh

for entry in os.scandir('/home/manish/Desktop/JhunAlignedWithBlob'):
    if entry.path.endswith('.jpg') and entry.is_file():
        src = entry.path
        start = src.find('image')+5
        end = src.find('.jpg')
        docId = src[start:end]
        imageOriginal = cv.imread(src, -1)
        imageThresh = image_thresholding(imageOriginal)
        imageSet = imageThresh[setCords[0]:setCords[1], setCords[2]:setCords[3]]
        imageStudent = imageThresh[studentCords[0]:studentCords[1], studentCords[2]:studentCords[3]]

        data = [docId, getIds(imageSet, 2), getIds(imageStudent, 9), getAnswerKeys(imageThresh)]
        df.loc[len(df.index)] = data
        lengi += 1
        print(f'Done processing image {lengi}: {docId}')
    #     if lengi == 1:
    #         break
    #     break
    # break

# print(df.head())
df['set_id'] = df['set_id'].apply('="{}"'.format)
df['student_id'] = df['student_id'].apply('="{}"'.format)
df['answer_keys'] = df['answer_keys'].apply('="{}"'.format)
df.to_csv('/home/manish/Desktop/bubble_processed3_V3.csv', index=False)
print(f'Total time taken: {(time.time() - start_time) / float(60)} minutes.')