import os
import cv2 as cv2
import numpy as np
import math
import mediapipe as mp

# auto balance brightness and contrast of image data
os.chdir("cuong gui\Vietnamese_hand_sign-main")
# hand detector tool
class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True):
        img_bone = np.ones((img.shape[0], img.shape[1], 3)) * 255
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        all_hands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                # lmList
                mylmList = []
                xList = []
                yList = []
                zList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                    zList.append(pz)
                # bbox
                x_min, x_max = min(xList), max(xList)
                y_min, y_max = min(yList), max(yList)
                boxW, boxH = x_max - x_min, y_max - y_min
                bbox = x_min, y_min, boxW, boxH

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["xList"] = xList
                myHand["yList"] = yList
                myHand["zList"] = zList
                all_hands.append(myHand)
                # draw
                if draw:
                    self.mpDraw.draw_landmarks(img_bone, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        if draw:
            return all_hands, img_bone
        else:
            return all_hands

    
# setup directory
folder_hand_path = "vietnamese_hand_sign\\classes temp" #folder temp contains cut videos
output_folder_frame_path = "vietnamese_hand_sign\\classes_frame_2"
output_folder_hand_path = "vietnamese_hand_sign\\classes_image_2" 
output_folder_bone_path = "vietnamese_hand_sign\\classes_bone_2"
output_folder_point_path = "vietnamese_hand_sign\\classes_point_2"
crop_frame = [(0.6, 0.5)]

for folder in os.listdir(folder_hand_path):
    # Create path
    child_folder_hand_path = os.path.join(folder_hand_path, folder)
    child_output_folder_frame_path = os.path.join(
        output_folder_frame_path, folder)
    child_output_folder_hand_path = os.path.join(
        output_folder_hand_path, folder)
    child_output_folder_bone_path = os.path.join(
        output_folder_bone_path, folder)
    child_output_folder_point_path = os.path.join(
        output_folder_point_path, folder)

    # Read file
    for index_video, file_video in enumerate(os.listdir(child_folder_hand_path)):
        video_hand_path = os.path.join(child_folder_hand_path, file_video)
        print(file_video)
        file_name = file_video.rstrip(".mp4")
        cap = cv2.VideoCapture(video_hand_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
        else:
            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(frame_total)
            detector = HandDetector(maxHands=1)
            offset = 35
            img_size = 300
            count = 0
            crop_x, crop_y = crop_frame[index_video]
            print(crop_x, crop_y)
            while count < (frame_total-10):
                ret, frame = cap.read()
                if frame is not None:
                    crop_x, crop_y = crop_frame[index_video]
                    print(frame.shape)
                    crop_x = crop_x * frame.shape[1] 
                    crop_y = crop_y * frame.shape[0]
                    print(crop_x, crop_y)
                    img = frame[0: int(crop_y), 0: int(crop_x)]
                    cv2.imwrite(os.path.join(child_output_folder_frame_path, f"{file_name}_{count}.jpg"), img)
                    
                    try:
                        hands, img_bone = detector.findHands(img)
                    except AttributeError:
                        pass
                    if hands:
                        hand = hands[0]
                        try:
                            # output points
                            lm_list = hand["lmList"]
                            x_list = hand["xList"]
                            y_list = hand["yList"]
                            z_list = hand["zList"]
                            x_min = min(x_list)
                            x_max = max(x_list)
                            y_min = min(y_list)
                            y_max = max(y_list)
                            z_min = min(z_list)
                            z_max = max(z_list)
                            
                            lm_list = np.array(lm_list)
                            lm_list_normalize = []
                            for lm in lm_list:
                                lm = [round((lm[0] - x_min) / (x_max - x_min), 2), round((lm[1] - y_min) / (y_max - y_min), 2),
                                    round(lm[2] / (z_max-z_min), 2)]
                                lm_list_normalize.append(lm)

                            output_point_name = file_name + "_" + str(count) + ".txt"
                            output_point_path = os.path.join(
                                child_output_folder_point_path, output_point_name)
                            np.savetxt(output_point_path,
                                    lm_list_normalize, fmt='%.2f')
                            # print("loi 1")
                            # output hand
                            x, y, w, h = hand['bbox']
                            
                            if h >= w:
                                img_hand_crop = frame[y - offset:y +
                                                    h + offset, x - offset:x + h + offset]
                            else:
                                img_hand_crop = frame[y - offset:y +
                                                    w + offset, x - offset:x + w + offset]

                            img_hand_resize = cv2.resize(
                                img_hand_crop, [img_size, img_size])
                            output_hand_name = file_name + "_" + str(count) + ".jpg"
                            output_hand_path = os.path.join(
                                child_output_folder_hand_path, output_hand_name)
                            cv2.imwrite(output_hand_path, img_hand_resize)
                            # print("loi 2")
                            # output bone
                            img_bone_crop = img_bone[y - offset:y +
                                                    h + offset, x - offset:x + w + offset]
                            img_white = np.ones(
                                (img_size, img_size, 3), np.uint8) * 255

                            aspectRatio = h / w
                            if aspectRatio > 1:
                                k = img_size / h
                                wCal = math.ceil(k * w)
                                img_bone_resize = cv2.resize(
                                    img_bone_crop, (wCal, img_size))
                                wGap = math.ceil((img_size - wCal) / 2)
                                img_white[:, wGap:wCal + wGap] = img_bone_resize
                            else:
                                k = img_size / w
                                hCal = math.ceil(k * h)
                                img_bone_resize = cv2.resize(
                                    img_bone_crop, (img_size, hCal))
                                hGap = math.ceil((img_size - hCal) / 2)
                                img_white[hGap:hCal + hGap, :] = img_bone_resize
                            output_bone_name = file_name + "_" + str(count) + ".jpg"
                            output_bone_path = os.path.join(
                                child_output_folder_bone_path, output_bone_name)
                            cv2.imwrite(output_bone_path, img_white)
                            # print("loi 3")
                            print(count)
                            count += 1
                            # cv2.imshow("ImageHand", img_hand_resize)
                            # cv2.imshow("ImageBone", img_white)
                            # key = cv2.waitKey(1)
                        except ValueError:
                            pass
                    else:
                        print("ko detect duoc tay")
                else:
                    count += 1
                    print(count, "loi")
                    # continue
            cap.release()
            print("done " + file_video)