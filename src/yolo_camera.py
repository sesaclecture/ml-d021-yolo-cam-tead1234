import torch
import cv2

# Model​
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

while True:


# Read frame (BGR to RGB)
    ret, frame = cap.read()
    if not ret:
        break

# 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)


    for i, obj in enumerate(results.xyxy[0]):
        conf = obj[4]
        obj_info = list(map(int, obj))
        print(f"Object {i}: {model.names[obj_info[5]]}")
        label = model.names[obj_info[5]]
        cv2.rectangle(frame, (obj_info[0], obj_info[1]), (obj_info[2], obj_info[3]), (0, 255, 0), 2)
        cv2.putText(frame, str(label) + str(conf), (obj_info[0], obj_info[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
    cv2.imshow("test", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()