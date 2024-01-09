# coding = utf-8
# @Time : 2023/12/6 22:36
# @Author : moyear
# @File : img_toushi_bianhuan.y
# @Software : PyCharm
import cv2
import numpy as np

# height, width, _ = img.shape

# 点击的坐标位置
p_list = []  # 左上，右上，左下，右下顺序点击

img2 = ""
img = ""

# todo 计算选择后的长宽
dst_rect = (200, 800)  # 变换目标大小, 是一个元组，第0个位置是宽长度，第1个位置是高长度


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    # 左上角点
    rect[0] = pts[np.argmin(s)]
    # 右下角点
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def auto_perspective_transform(image, pts):
    # todo 这个方法是根据四个角点的位置直接算出新的透射变化后的图形的长与宽，
    #  但是这种方法可能与实际坡面的长宽比不一致
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def four_point_transform(img, rect, dst_rect):
    # todo 输入一个图像，并指定四个角点的位置，然后再根据需要透射变换后的目标图像的长宽进行透射变换
    # 透射变换后的4个角点顺序：左上、右上、右下、左下
    pts2 = np.float32([[0, 0], [dst_rect[0], 0], [dst_rect[0], dst_rect[1]], [0, dst_rect[1]]])

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, pts2)
    dst = cv2.warpPerspective(img, M, dst_rect)
    return dst


def capture_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 绘制点击的位置
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("original_img", img)
        p_list.append([x, y])

        if len(p_list) == 4:
            # 对选择的4个角点进行排序，顺序：左上、右上、右下、左下
            rect = order_points(np.float32(p_list))

            img_dst = four_point_transform(img2, rect, dst_rect)  # 按照指定的长宽进行透射变换
            # img_dst = auto_perspective_transform(img2, rect)  # 根据选择的角点位置，自动计算变换后的长宽，然后进行透射变换

            print("after perspective transformation, image height: {0}, width: {1}"
                  .format(img_dst.shape[0], img_dst.shape[1]))

            # _, img_dst = cv2.threshold(img_dst[:,:,2], 127, 255, cv2.THRESH_BINARY)

            cv2.imshow("result_img", img_dst)
            cv2.imwrite('../output/transformed.jpg', img_dst)  # 输出图像到文件


def show_and_transform(img_path):
    global img, img2

    img = cv2.imread(img_path)  # 输入图像
    img2 = img.copy()

    cv2.namedWindow("original_img")
    cv2.imshow("original_img", img)
    cv2.setMouseCallback("original_img", capture_event)
    cv2.waitKey(0)


def show_and_transform(img_path):
    global img, img2

    img = cv2.imread(img_path)  # 输入图像
    img2 = img.copy()

    cv2.namedWindow("original_img")
    cv2.imshow("original_img", img)
    cv2.setMouseCallback("original_img", capture_event)
    cv2.waitKey(0)


def show_raw(raw_path):
    ...