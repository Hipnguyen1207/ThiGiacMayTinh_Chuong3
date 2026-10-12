# ThiGiacMayTinh_Chuong3

## Bai 1: Webcam, ảnh xám và nhận diện

Chương trình `Bai1/bai1.py` mở webcam, hiển thị ảnh xám và các thông số FPS, độ phân giải, số khuôn mặt. OpenCV 4 dùng Haar Cascade; OpenCV 5 dùng YuNet và tự tải model vào cache ở lần chạy đầu. YOLO là tùy chọn để nhận diện vật thể; khuôn mặt được nhận diện riêng bằng OpenCV.

Cài OpenCV nếu môi trường Python chưa có:

```powershell
python -m pip install opencv-python
```

Chạy chế độ webcam và nhận diện khuôn mặt:

```powershell
python Bai1/bai1.py
```

Cài YOLO và dùng model nano hoặc small:

```powershell
python -m pip install ultralytics
python Bai1/bai1.py --yolo nano
python Bai1/bai1.py --yolo small
```

Lần chạy YOLO đầu tiên sẽ tải trọng số model. Thiết bị mặc định là CUDA nếu PyTorch nhận diện được GPU NVIDIA tương thích, nếu không sẽ dùng CPU. Để dùng GPU, hãy cài bản PyTorch phù hợp với hệ thống CUDA của máy theo hướng dẫn chính thức của PyTorch. Có thể chọn thiết bị bằng `--device cuda` hoặc `--device cpu`.

Các tùy chọn khác: `--camera 1` chọn webcam khác, `--width 640 --height 480` đổi độ phân giải, `--confidence 0.5` đổi ngưỡng tin cậy. Nhấn `Q` hoặc `Esc` để thoát.

## Bai 2: Webcam và tệp ảnh/video

Chạy giao diện hai tab bằng:

```powershell
python Bai2/bai.py
```

Tab `Webcam` giữ màu gốc, nhận diện mặt và YOLO. Tab `File` cho phép chọn ảnh hoặc video từ ổ đĩa; video có nút phát/tạm dừng. Cả hai tab đều giữ màu gốc. Nhấn `A` để xoay 90 độ ngược chiều kim đồng hồ hoặc `D` để xoay 90 độ theo chiều kim đồng hồ. Có thể chọn YOLO `nano` hoặc `small` ở thanh trên.

## Bai 3: Lưu kết quả nhận diện

Chạy giao diện bằng `python Bai3/bai3`. Chọn ảnh hoặc video trong tab `File`; chương trình nhận diện mặt trên ảnh hoặc từng frame video. Nhấn `Ctrl+S` để lưu ảnh kết quả đang hiển thị (có khung nhận diện và thông số). Với video, phím tắt lưu frame hiện tại thành ảnh, không xuất toàn bộ video.

## Bai 4: Đếm bóng

Chạy giao diện bằng `python Bai4/bai4.py`. Hai tab `Webcam` và `File` được giữ nguyên; tab `Đếm bóng` cho phép chọn ảnh từ ổ đĩa, kết hợp contour có lọc độ tròn với HoughCircles để đếm bóng, kể cả một số bóng nằm sát nhau. Thanh `Độ nhạy` điều chỉnh bộ lọc và ngưỡng phát hiện.

## Bai 5: Tách ký tự biển số

Chạy `python Bai5/bai5.py`, chọn ảnh xe hoặc ảnh biển số. Với ảnh toàn xe, dùng `Khoanh vùng biển số` rồi kéo chọn ROI; với ảnh biển số đã cắt sẵn, chọn `Dùng toàn ảnh`. Nhấn `Tách ký tự` để xem các crop, sau đó nhấn `Lưu PNG` hoặc `Ctrl+S` để xuất từng ký tự thành một tệp riêng trong thư mục `<tên ảnh>_characters`.