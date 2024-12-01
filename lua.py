import cv2
from ultralytics import YOLO

# Tải mô hình YOLOv8 đã huấn luyện
model = YOLO('lua.pt')  # Đường dẫn đến mô hình lua.pt

# Đường dẫn video
video_path = 'chay.mp4'

# Mở video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Thực hiện phát hiện đối tượng
    results = model(frame)

    # Vẽ bounding box cho các đối tượng phát hiện
    for result in results:
        boxes = result.boxes  # Lấy thông tin hộp
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Tọa độ hộp bao
            conf = box.conf[0]  # Độ tin cậy
            cls = box.cls[0]  # Lớp đối tượng
            label = f'{model.names[int(cls)]} {conf:.2f}'  # Nhãn và độ tin cậy
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Vẽ hình chữ nhật
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Ghi nhãn

    # Hiển thị khung hình
    cv2.imshow('Fire and Smoke Detection', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
