import more_SS
import cv2

if __name__ == '__main__':
    filePath= './images/'
    savePath= './result/'
    fileName= more_SS.get_file_name(filePath)
    for name in fileName:
        img = cv2.imread(filePath+ name)
        region = more_SS.selective_search(img, 0.9, 1000, 10)
        img = more_SS.rect_draw(img, region, color= (0, 0, 255), regionSize= 2000)
        cv2.imwrite(savePath+ 'SS_'+ name, img)
        print("Image {} is OK.".format(name))
