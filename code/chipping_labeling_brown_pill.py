import json
import glob
import cv2 as cv
import numpy as np
import shutil


def display_img(image, mask):
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    new_mask = np.bitwise_and(image, mask)
    inv_reg = cv.bitwise_not(mask)
    n_image = np.bitwise_and(image, inv_reg)
    mask[:, :, 0] = mask[:, :, 0]/255 * 249
    mask[:, :, 1] = mask[:, :, 1]/255 * 134
    mask[:, :, 2] = mask[:, :, 2]/255 * 31
    new_mask = cv.addWeighted(new_mask, 0.6, mask, 0.4, 0)
    n_image = n_image + new_mask
    return n_image


file_list = glob.glob('chipping_test/*.png')
m = 0
for filename in file_list:
    pionts = {}
    shapes = []
    f_name, ext = filename.split('.')
    img = cv.imread(filename, 1)
    # img = cv.resize(img, (384, 384))
    b, g, r = cv.split(img)
    r = cv.medianBlur(r, 5)
    #ret, r_mask = cv.threshold(r, 70, 255, cv.THRESH_BINARY)
    ret, r_mask = cv.threshold(r, 30, 255, cv.THRESH_BINARY)
    # r_mask = cv.dilate(r_mask, kernel=np.ones((1, 1), np.uint8))
    b = cv.bitwise_and(b, r_mask)
    b_mask = cv.medianBlur(b, 15)
    b = cv.medianBlur(b, 5)
    # b = cv.blur(b, (3, 3))
    # cv.imshow('test', b)
    # cv.waitKey()
    b = cv.subtract(b, b_mask)*3
    #low_gv = 40
    #k_erode = 5
    #k_dilate = 5
    low_gv = 9
    k_erode = 5
    k_dilate = 5

    while True:
        b_mask = cv.blur(b_mask, (5, 5))
        b_mask = cv.inRange(b, low_gv, 50)
        a = np.zeros_like(b_mask)
        contours, hierarchy = cv.findContours(b_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            area = cv.contourArea(contours[i])
            if area < 40:
                cv.drawContours(b_mask, contours, i, 0, cv.FILLED)

        b_mask = cv.dilate(b_mask, kernel=np.ones((k_dilate, k_dilate), np.uint8))
        b_mask = cv.erode(b_mask, kernel=np.ones((k_erode, k_erode), np.uint8))
        new_image = display_img(img, b_mask)
        new_image = cv.hconcat([cv.resize(img, (768, 768)), cv.resize(new_image, (768, 768))])
        cv.imshow(filename, new_image)
        cv.moveWindow(filename, 10, 10)
        k = cv.waitKey(0)
        cv.destroyAllWindows()
        if k == 27:  # ESC key
            break
        elif k == 13:  # Return Key
            if m < 10:
                save_string = 'chipping_test/white_chipping_000' + str(m)
            elif m < 100:
                save_string = 'chipping_test/white_chipping_00' + str(m)
            elif m < 1000:
                save_string = 'chipping_test/white_chipping_0' + str(m)
            elif m < 10000:
                save_string = 'chipping_test/white_chipping_' + str(m)
            cv.imwrite(save_string + '.png', img)
            cv.imwrite(save_string + '_gt.png', b_mask)
            json_f_name = save_string.split('/')
            contours, hierarchy = cv.findContours(b_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                shape = {"label": "chipping", "points": np.squeeze(contours[i], axis=1).tolist(), "group_id": None,
                         "shape_type": "polygon", "flag": {}
                         }
                shapes.append(shape)
            # print(i, shapes)
            j_data = {"version": "4.2.9", "flags": {}, "shapes": shapes, "imagePath": json_f_name[-1]+'.png', "imageData": None,  "imageHeight": 768, "imageWidth": 768}
            with open(save_string + '.json', 'w') as fp:
                json.dump(j_data, fp, indent=2)
                fp.close()
            m += 1
            break
        elif k == 105:  # i key (increase threshold gv)
            low_gv += 2
            print('gv+5', low_gv)
        elif k == 100:  #  d key (decrease threshold gv)
            low_gv -= 2
            print('gv-5', low_gv)
        elif k == 112:  # p key (move image to picking folder)
            f_path, name = f_name.split('\\')
            shutil.move(filename, f_path + '/picking/'+name+'.png')
            cv.imwrite(f_path + '/picking/' + name + '_gt.png', b_mask)
            break
        elif k == 103:  # g key (move image to good folder)
            f_path, name = f_name.split('\\')
            shutil.move(filename, f_path + '/good/'+name+'.png')
            break
        elif k == 101:  # e key (increase kernel size of erosion)
            k_erode += 1
            print('kernel size of erode = ', k_erode)
        elif k == 119:  # w key (decrease kernel size of erosion)
            k_erode -= 1
            print('kernel size of erode = ', k_erode)






    # print(name)