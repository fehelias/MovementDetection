import cv2
import settings

while True:
    #reading video
    RET_OBJECT, FRAME_OBJECT = settings.INPUT_VIDEO.read()
    RET_TRAIN1, FRAME_TRAIN1 = settings.INPUT_VIDEO.read()
    RET_TRAIN2, FRAME_TRAIN2 = settings.INPUT_VIDEO.read()
    RET_TRAIN3, FRAME_TRAIN3 = settings.INPUT_VIDEO.read()

    #setting height and width
    HEIGHT_OBJECT, WIDTH_OBJECT, _ = FRAME_OBJECT.shape
    HEIGHT_TRAIN1, WIDTH_TRAIN1, _ = FRAME_TRAIN1.shape
    HEIGHT_TRAIN2, WIDTH_TRAIN2, _ = FRAME_TRAIN2.shape
    HEIGHT_TRAIN3, WIDTH_TRAIN3, _ = FRAME_TRAIN3.shape

    #setting the area
    ROI_OBJECT = FRAME_OBJECT[160:720, 100:1000]
    ROI_TRAIN1 = FRAME_TRAIN1[400:, :400]
    ROI_TRAIN2 = FRAME_TRAIN2[500:, :300]
    ROI_TRAIN3 = FRAME_TRAIN3[70:85, 400:500]

    #creating mask
    MASK_OBJECT = settings.CARS_DETECTION.apply(ROI_OBJECT)
    MASK_TRAIN1 = settings.TRAIN_DETECTION1.apply(ROI_TRAIN1)
    MASK_TRAIN2 = settings.TRAIN_DETECTION2.apply(ROI_TRAIN2)
    MASK_TRAIN3 = settings.TRAIN_DETECTION3.apply(ROI_TRAIN3)

    #binary transformation
    _, MASK_OBJECT = cv2.threshold(MASK_OBJECT, 253, 254, cv2.THRESH_BINARY)
    _, MASK_TRAIN1 = cv2.threshold(MASK_TRAIN1, 253, 254, cv2.THRESH_BINARY)
    _, MASK_TRAIN2 = cv2.threshold(MASK_TRAIN2, 253, 254, cv2.THRESH_BINARY)
    _, MASK_TRAIN3 = cv2.threshold(MASK_TRAIN3, 253, 254, cv2.THRESH_BINARY)

    #creating contours
    CONTOURS_OBJECT, _ = cv2.findContours(MASK_OBJECT, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    CONTOURS_TRAIN1, _ = cv2.findContours(MASK_TRAIN1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    CONTOURS_TRAIN2, _ = cv2.findContours(MASK_TRAIN2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    CONTOURS_TRAIN3, _ = cv2.findContours(MASK_TRAIN3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #setting conditions for printing
    for cnt_obj in CONTOURS_OBJECT:
        area_object = cv2.contourArea(cnt_obj)

        if area_object > 100:
            cv2.putText(FRAME_OBJECT, "Status: {}".format("Objeto passando"), (0, 25), cv2.FONT_HERSHEY_TRIPLEX, 
                                1, (255, 0, 0), 2)

    for cnt_train1 in CONTOURS_TRAIN1:
        area_train1 = cv2.contourArea(cnt_train1)

        if area_train1 > 30 and area_train2 > 30:
            print("Trem passando")

    for cnt_train2 in CONTOURS_TRAIN2:
        area_train2 = cv2.contourArea(cnt_train2)

        if area_train2 > 30 and area_train1 > 30:
            print("Trem passando")

    for cnt_train3 in CONTOURS_TRAIN3:
        area_train3 = cv2.contourArea(cnt_train3)

        if area_train3 > 30:
            print("Trem passando")

    #display video
    cv2.imshow("OBJECT DETECTOR", FRAME_OBJECT)

    key = cv2.waitKey(30)

    if key == 27:
        break

settings.INPUT_VIDEO.release()
cv2.destroyAllWindows()