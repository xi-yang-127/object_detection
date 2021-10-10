# coding="utf-8"
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import imutils

# 初始化方向梯度直方图描述子
hog = cv2.HOGDescriptor()
# 设置支持向量机使得它成为一个预先训练好了的行人检测器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# 读取摄像头视频
cap = cv2.VideoCapture(0)
total = 0
while True:
    # 按帧读取视频
    ret, img = cap.read()
    # 将每一帧图像ROI区域抠出来
    # img_roi = img1[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]
    # roi = img[50:450, 350:600]
    # 通过调用detectMultiScale的hog描述子方法，对图像中的行人进行检测。
    (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.03)# 1.05
    # 应用非极大值抑制，通过设置一个阈值来抑制重叠的框。
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    print(pick)
    # 绘制红色人体矩形框
    for (x, y, w, h) in pick:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 绘制绿色对比框
    # cv2.rectangle(img, (50, 50), (600, 450), (0, 255, 0), 2)
    # 绘制蓝色危险区域框
    cv2.rectangle(img, (50, 50), (600, 450), (255, 0, 0), 2)
    # 打印检测到的目标个数
    total += len(pick)
    print(f"行人个数为{len(pick)}, 总计：{total}")
    # 展示每一帧图像
    cv2.imshow("HOG+SVM+NMS", img)
    # 按esc键退出循环
    if cv2.waitKey(1) & 0xff == 27:
        break
