import cv2
import face_recognition
import numpy as np
import os
import datetime

assets = {
    'glasses': [None, 'sunglasses.png', 'sunglassesblue.png', 'pixelatedglasses.png'],
    'headgears': [None, 'baseballcap.png', 'magichat.png', 'greencrown.png', 'flowersbouquet.png'],
    'masks': [None, 'dogtongue.png']
}
headgear, glasses, mask = None, None, None

def box_contains_point(bbox, pt):
    return bbox[0] < pt[0] < bbox[0] + bbox[2] and bbox[1] < pt[1] < bbox[1] + bbox[3]

def bounding_box_is_inside_frame(bbox, frame_w, frame_h):
    frame_bbox = (0, 0, frame_w, frame_h)
    bbox_points = [
        (bbox[0], bbox[1]),
        (bbox[0] + bbox[2], bbox[1]),
        (bbox[0] + bbox[2], bbox[1] + bbox[3]),
        (bbox[0], bbox[1] + bbox[3])
    ]
    for point in bbox_points:
        if not box_contains_point(frame_bbox, point):
            return False
    return True

def next_item(asset, current_item):
    current_item_index = assets[asset].index(current_item)
    if current_item_index == len(assets[asset]) - 1:
        return assets[asset][0]
    else:
        return assets[asset][current_item_index + 1]

def wear(frame, item_name, item_bounding_box):
    frame_h, frame_w, _ = frame.shape
    if bounding_box_is_inside_frame(item_bounding_box, frame_w, frame_h):
        x, y, w, h = item_bounding_box
        item_roi = frame[y:y + h, x:x + w] # get region where item is to be placed
        item = cv2.imread('./assets/' + item_name, cv2.IMREAD_UNCHANGED)
        resized_item = cv2.resize(item, (w, h)) # resize image to fit roi
        a = cv2.split(resized_item)[3] # get alpha channel of item
        _mask = cv2.merge((a, a, a)) # create mask from alpha channel
        # alpha blend
        fg = cv2.cvtColor(resized_item, cv2.COLOR_BGRA2BGR).astype(float)
        bg = item_roi.astype(float)
        alpha = _mask.astype(float) / 255
        fg = cv2.multiply(alpha, fg)
        bg = cv2.multiply(1.0 - alpha, bg)
        blend = cv2.add(fg, bg)
        # place blend in frame
        frame[y:y + h, x:x + w] = blend
    return frame

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    resized_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # scale frame to 1/4 size for faster processing
    resized_frame_rgb = resized_frame[:, :, ::-1] # convert frame from BGR (OpenCV) to RGB (face_recognition)
    face_locations = face_recognition.face_locations(resized_frame_rgb)  # find all faces using HOG-based mode

    for face_location in face_locations:
        top, right, bottom, left = np.array(face_location) * 4 # scale face loc back up
        face_width = right - left
        face_height = bottom - top

        if glasses != None:
            # assumption: if you divide height of face in to 6 parts, eyes are contained within 2nd & 3rd parts (from top)
            glasses_bbox = (left, # x
                            int(top + round(1/6 * face_height)),
                            face_width,
                            int(round(3/6 * face_height - 1/6 * face_height)))
            frame = wear(frame, glasses, glasses_bbox)

        if headgear != None:
            # assumption: headgear height is 2/3 of face height
            headgear_bbox = (left,
                            int(top - round(2/3 * face_height)),
                            face_width,
                            int(round(2/3 * face_height)))
            frame = wear(frame, headgear, headgear_bbox)

        if mask != None:
            # assumption: mask covers the whole face
            mask_bbox = (left, top, face_width, face_height)
            frame = wear(frame, mask, mask_bbox)

        # draw face bounding box
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 225, 0), 2)

    cv2.imshow('Snapclone', frame)

    k = cv2.waitKey(1) & 0xFF
    # toggle through different items
    if k == ord('1'):
        glasses = next_item('glasses', glasses)
    if k == ord('2'):
        headgear = next_item('headgears', headgear)
    if k == ord('3'):
        mask = next_item('masks', mask)
    # save photo if 's' key is pressed
    if k & 0xFF == ord('s'):
        cv2.imwrite(os.path.join('photos', 'photo_' + str(datetime.datetime.now()) + '.png'), frame)
    # quit camera if 'q' key is pressed
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()