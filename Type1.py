import pandas as pd
import skimage.io as io
import glob
from PIL import Image
import cv2
import shutil
import numpy as np
class Data(object):
    def __init__(self):
        self.label = {}
        df = pd.read_csv('sample_data/sample_submit_type_1.csv')
        for i, row in df.iterrows():
            self.label[row.IMG_ID] = row.GUESS
        print(self.label['0a0c7b9c-4e7b-11ea-a0ad-001a7dda7113'])
        print(len(self.label))
        dir = 'sample_data\\Type_1'
        for file in glob.glob(f'{dir}\\*.jpg'):
            id = file.split('\\')[-1].split('.')[0]
            if id not in self.label:
                print(id)
                shutil.copy(file, f'sample_data\\test\\{id}.jpg')
            #
        # self.load_imgs('sample_data/Type_1')

    def load_imgs(self, dir):
        for file in glob.glob(f'{dir}/*.jpg'):
            img = Image.open(file)
            print(img)



if __name__ == '__main__':
    # Data()
    # %%
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    img = cv2.imread('sample_data/Type_1/0af7141e-4e7b-11ea-8c8b-001a7dda7113.jpg',cv2.IMREAD_COLOR)
    # img = cv2.imread('sample_data/Type_1/0af7141e-4e7b-11ea-8c8b-001a7dda7113.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('sample_data/Type_1/0b1c83e6-4e7b-11ea-a116-001a7dda7113.jpg', 0)
    laplacian = cv2.Laplacian(img, -1)
    laplacian = cv2.convertScaleAbs(laplacian)/255
    laplacian = np.mean(np.square(laplacian), axis=-1)
    laplacian = laplacian/np.max(laplacian)
    plt.imshow(laplacian)
    plt.show()
    exit()
    # plt.imshow(img)
    # plt.show()

    # erosion = img
    # k = 3
    # kernel = np.ones((k, k), np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # plt.imshow(np.round(erosion).astype(np.uint8))
    # plt.show()

    blur = cv2.bilateralFilter(img, 7, 89, 50)
    thres, binary = cv2.threshold(blur, 82, 255, cv2.THRESH_BINARY_INV)

    # plt.imshow(binary)
    # res = cv2.blur(binary, (3, 3))
    # thres, binary = cv2.threshold(res, 200, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                       cv2.THRESH_BINARY_INV, 11, 2)
    plt.imshow(binary)
    plt.show()
    exit()

    blur = cv2.bilateralFilter(img, 9, 75, 75)
    plt.imshow(blur)
    plt.show()

    thres, binary = cv2.threshold(blur, 82, 255, cv2.THRESH_BINARY_INV)
    # binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                       cv2.THRESH_BINARY_INV, 11, 2)
    plt.imshow(cv2.blur(img, ksize=5))
    plt.show()
    exit()

    img = binary
    # k = 2
    # dilate_kernel = np.ones((k, k), np.uint8)/k**2
    # openning = cv2.dilate(img, dilate_kernel, iterations=1)
    # plt.imshow(openning)
    # plt.show()


    # def doCanny(x):
    #     # gauss = cv2.GaussianBlur(img, (3, 3), 1)
    #     position = cv2.getTrackbarPos("CannyBar", "Canny")
    #     # canny = cv2.Canny(binary, position, position * 5)
    #     canny = cv2.Laplacian(binary, ddepth=-1)
    #     cv2.imshow("Canny", canny)
    # cv2.namedWindow("Canny")
    # cv2.createTrackbar("CannyBar", "Canny", 1, 255, doCanny)
    # cv2.waitKey(0)
    # exit()






    # def do_thres(x):
    #     # gauss = cv2.GaussianBlur(img, (3, 3), 1)
    #     position = cv2.getTrackbarPos("ThresholdBar", "Canny")*255/100
    #     thres, binary = cv2.threshold(blur, position, maxval=255, type=cv2.THRESH_BINARY_INV)
    #     # canny = cv2.Canny(blur, position, position * 1.5)
    #     cv2.imshow("Canny", binary)
    # cv2.namedWindow("Canny")
    # cv2.createTrackbar("ThresholdBar", "Canny", 1, 100, do_thres)
    # cv2.waitKey(0)

    # canny = cv2.Canny(blur, 32*255/100, 32*255/100 * 1.5)
    # plt.imshow(canny)
    # plt.show()