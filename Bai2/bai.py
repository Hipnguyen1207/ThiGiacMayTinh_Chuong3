"""Live webcam preview with grayscale display, face detection, and optional YOLO."""

import argparse
import importlib
import os
import tkinter as tk
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image, ImageTk


YOLO_MODELS = {
	"nano": "yolov8n.pt",
	"small": "yolov8s.pt",
}
YUNET_MODEL_URL = (
	"https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
	"models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Webcam grayscale preview with face detection and optional YOLO."
	)
	parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
	parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
	parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
	parser.add_argument(
		"--yolo",
		choices=YOLO_MODELS,
		help="Enable YOLO using the nano or small model (downloads weights on first use).",
	)
	parser.add_argument(
		"--device",
		choices=("auto", "cpu", "cuda"),
		default="auto",
		help="YOLO processing device; auto selects CUDA when available.",
	)
	parser.add_argument(
		"--confidence", type=float, default=0.35, help="YOLO confidence threshold (0-1)."
	)
	return parser.parse_args()


def select_device(requested_device):
	if requested_device == "cpu":
		return "cpu"

	try:
		torch = importlib.import_module("torch")
	except ImportError:
		if requested_device == "cuda":
			raise RuntimeError("CUDA requested, but PyTorch is not installed.") from None
		return "cpu"

	if torch.cuda.is_available():
		return "cuda:0"
	if requested_device == "cuda":
		raise RuntimeError("CUDA requested, but no CUDA-enabled GPU is available.")
	return "cpu"


def create_yolo(model_size, device):
	try:
		YOLO = importlib.import_module("ultralytics").YOLO
	except ImportError as error:
		raise RuntimeError(
			"YOLO is unavailable. Install it with: pip install ultralytics"
		) from error

	model = YOLO(YOLO_MODELS[model_size])
	return model, device


def create_face_detector(frame_width, frame_height):
	if hasattr(cv2, "CascadeClassifier"):
		cascade = cv2.CascadeClassifier(
			cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
		)
		if cascade.empty():
			raise RuntimeError("OpenCV could not load its Haar face-detection model.")
		return cascade, None

	if not hasattr(cv2, "FaceDetectorYN_create"):
		raise RuntimeError(
			"This OpenCV build has no supported face detector. "
			"Install OpenCV 4 with: python -m pip install --force-reinstall 'opencv-python<5'"
		)

	cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
	model_path = cache_root / "opencv_models" / "face_detection_yunet_2023mar.onnx"
	if not model_path.exists():
		model_path.parent.mkdir(parents=True, exist_ok=True)
		temporary_path = model_path.with_suffix(".onnx.download")
		try:
			urlretrieve(YUNET_MODEL_URL, temporary_path)
			temporary_path.replace(model_path)
		except OSError as error:
			temporary_path.unlink(missing_ok=True)
			raise RuntimeError(
				"Could not download the YuNet face model. Check your internet connection "
				"or install OpenCV 4 with: python -m pip install --force-reinstall 'opencv-python<5'"
			) from error

	detector = cv2.FaceDetectorYN_create(
		str(model_path), "", (frame_width, frame_height)
	)
	return None, detector


def detect_faces(frame, gray, cascade, yunet_detector):
	if cascade is not None:
		return cascade.detectMultiScale(
			gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
		)

	yunet_detector.setInputSize((frame.shape[1], frame.shape[0]))
	_, detections = yunet_detector.detect(frame)
	if detections is None:
		return []
	return [tuple(int(value) for value in detection[:4]) for detection in detections]


def draw_yolo_detections(image, result):
	if result.boxes is None:
		return

	for box in result.boxes:
		x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
		class_id = int(box.cls[0].item())
		confidence = float(box.conf[0].item())
		label = result.names[class_id]
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 190, 255), 2)
		cv2.putText(
			image,
			f"{label} {confidence:.0%}",
			(x1, max(22, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(0, 190, 255),
			2,
			cv2.LINE_AA,
		)


class WebcamFileApp:
	IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

	def __init__(self, root, args):
		self.root = root
		self.args = args
		self.root.title("OpenCV | Webcam va File")
		self.root.geometry("1100x760")
		self.root.minsize(760, 560)
		self.root.configure(bg="#171b1f")
		self.root.protocol("WM_DELETE_WINDOW", self.close)

		self.camera = None
		self.file_capture = None
		self.file_path = None
		self.file_kind = None
		self.file_frame = None
		self.video_playing = False
		self.rotation = 0
		self.yolo_model = None
		self.yolo_device = "off"
		self.last_frame_time = time.perf_counter()
		self.fps = 0.0
		self.photo = None
		self.face_cascade, self.yunet_detector = create_face_detector(
			args.width, args.height
		)

		self._configure_style()
		self._build_interface()
		if args.yolo:
			self.model_var.set(args.yolo)
			self._load_yolo(args.yolo, show_errors=False)

		self.root.bind_all("<KeyPress>", self._on_key)
		self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
		self._start_webcam()
		self.root.after(20, self._update_frame)

	def _configure_style(self):
		style = ttk.Style(self.root)
		style.theme_use("clam")
		style.configure("TFrame", background="#171b1f")
		style.configure("TLabel", background="#171b1f", foreground="#e9eef2")
		style.configure("TButton", padding=(12, 7))
		style.configure(
			"TNotebook", background="#171b1f", borderwidth=0
		)
		style.configure(
			"TNotebook.Tab", background="#252c32", foreground="#bac5cc", padding=(22, 10)
		)
		style.map(
			"TNotebook.Tab",
			background=[("selected", "#287c78")],
			foreground=[("selected", "#ffffff")],
		)

	def _build_interface(self):
		header = ttk.Frame(self.root, padding=(18, 14))
		header.pack(fill="x")
		ttk.Label(
			header, text="LIVE IMAGE LAB", font=("Segoe UI Semibold", 15)
		).pack(side="left")
		ttk.Label(header, text="YOLO model").pack(side="left", padx=(28, 8))
		self.model_var = tk.StringVar(value="off")
		self.model_picker = ttk.Combobox(
			header,
			textvariable=self.model_var,
			values=("off", "nano", "small"),
			state="readonly",
			width=10,
		)
		self.model_picker.pack(side="left")
		self.model_picker.bind("<<ComboboxSelected>>", self._on_model_changed)
		self.rotation_label = ttk.Label(header, text="Rotation: 0°")
		self.rotation_label.pack(side="right")

		self.notebook = ttk.Notebook(self.root)
		self.notebook.pack(fill="both", expand=True, padx=16, pady=(0, 16))
		self.webcam_tab = ttk.Frame(self.notebook, padding=10)
		self.file_tab = ttk.Frame(self.notebook, padding=10)
		self.notebook.add(self.webcam_tab, text="Webcam")
		self.notebook.add(self.file_tab, text="File")

		self.webcam_view = tk.Label(
			self.webcam_tab,
			text="Đang kết nối webcam...",
			bg="#090c0f",
			fg="#bac5cc",
			font=("Segoe UI", 13),
		)
		self.webcam_view.pack(fill="both", expand=True)
		self.webcam_status = ttk.Label(self.webcam_tab, text="Webcam")
		self.webcam_status.pack(anchor="w", pady=(8, 0))

		file_controls = ttk.Frame(self.file_tab)
		file_controls.pack(fill="x", pady=(0, 10))
		ttk.Button(
			file_controls, text="Chọn ảnh / video", command=self._choose_file
		).pack(side="left")
		self.play_button = ttk.Button(
			file_controls,
			text="Tạm dừng",
			command=self._toggle_video,
			state="disabled",
		)
		self.play_button.pack(side="left", padx=(8, 0))
		self.file_status = ttk.Label(file_controls, text="Chưa chọn tệp")
		self.file_status.pack(side="left", padx=12)

		self.file_view = tk.Label(
			self.file_tab,
			text="Chọn một ảnh hoặc video để bắt đầu",
			bg="#090c0f",
			fg="#bac5cc",
			font=("Segoe UI", 13),
		)
		self.file_view.pack(fill="both", expand=True)

	def _start_webcam(self):
		if self.camera is not None and self.camera.isOpened():
			return
		self.camera = cv2.VideoCapture(self.args.camera, cv2.CAP_DSHOW)
		if not self.camera.isOpened():
			self.camera.release()
			self.camera = cv2.VideoCapture(self.args.camera)
		if not self.camera.isOpened():
			self.camera.release()
			self.camera = None
			self.webcam_status.configure(text="Không thể mở webcam")
			self.webcam_view.configure(text="Không thể mở webcam", image="")
			return
		self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
		self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
		self.webcam_status.configure(text=f"Webcam {self.args.camera} đang hoạt động")

	def _stop_webcam(self):
		if self.camera is not None:
			self.camera.release()
			self.camera = None

	def _stop_file_video(self):
		if self.file_capture is not None:
			self.file_capture.release()
			self.file_capture = None
		self.video_playing = False

	def _choose_file(self):
		path = filedialog.askopenfilename(
			parent=self.root,
			title="Chọn ảnh hoặc video",
			filetypes=(
				("Ảnh và video", "*.bmp *.jpeg *.jpg *.mkv *.mov *.mp4 *.png *.tif *.tiff *.avi *.webm *.wmv"),
				("Tất cả tệp", "*.*"),
			),
		)
		if path:
			self._load_file(path)

	def _load_file(self, path, show_errors=True, reset_rotation=True):
		self._stop_file_video()
		self.file_path = path
		self.file_frame = None
		self.file_kind = None
		if reset_rotation:
			self.rotation = 0
			self._update_rotation_label()

		extension = Path(path).suffix.lower()
		if extension in self.IMAGE_EXTENSIONS:
			try:
				encoded = np.fromfile(path, dtype=np.uint8)
				self.file_frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
			except (OSError, cv2.error) as error:
				self.file_frame = None
				if show_errors:
					messagebox.showerror("Không đọc được ảnh", str(error), parent=self.root)
				return
			if self.file_frame is None:
				if show_errors:
					messagebox.showerror(
						"Không đọc được ảnh", "Định dạng ảnh không được hỗ trợ.", parent=self.root
					)
				return
			self.file_kind = "image"
			self.file_status.configure(text=Path(path).name)
			self.play_button.configure(state="disabled", text="Tạm dừng")
			return

		capture = cv2.VideoCapture(path)
		if not capture.isOpened():
			capture.release()
			if show_errors:
				messagebox.showerror(
					"Không đọc được video", "Không thể mở tệp video đã chọn.", parent=self.root
				)
			return
		success, frame = capture.read()
		if not success:
			capture.release()
			if show_errors:
				messagebox.showerror(
					"Video trống", "Không đọc được khung hình đầu tiên.", parent=self.root
				)
			return
		self.file_capture = capture
		self.file_frame = frame
		self.file_kind = "video"
		self.video_playing = True
		self.video_fps = capture.get(cv2.CAP_PROP_FPS)
		if not 1 <= self.video_fps <= 120:
			self.video_fps = 30
		self.file_status.configure(text=Path(path).name)
		self.play_button.configure(state="normal", text="Tạm dừng")

	def _toggle_video(self):
		if self.file_kind != "video" or self.file_capture is None:
			return
		self.video_playing = not self.video_playing
		self.play_button.configure(
			text="Tạm dừng" if self.video_playing else "Phát"
		)

	def _on_model_changed(self, _event=None):
		model_name = self.model_var.get()
		if model_name == "off":
			self.yolo_model = None
			self.yolo_device = "off"
			return
		self._load_yolo(model_name)

	def _load_yolo(self, model_name, show_errors=True):
		try:
			device = select_device(self.args.device)
			self.yolo_model, self.yolo_device = create_yolo(model_name, device)
		except (RuntimeError, OSError) as error:
			self.yolo_model = None
			self.yolo_device = "off"
			self.model_var.set("off")
			if show_errors:
				messagebox.showerror("Không thể bật YOLO", str(error), parent=self.root)

	def _on_tab_changed(self, _event=None):
		selected = self.notebook.tab(self.notebook.select(), "text")
		if selected == "Webcam":
			self._stop_file_video()
			if self.file_kind == "video":
				self.file_frame = None
			self._start_webcam()
		else:
			self._stop_webcam()
			if self.file_kind == "video" and self.file_capture is None and self.file_path:
				self._load_file(self.file_path, show_errors=False, reset_rotation=False)

	def _on_key(self, event):
		key = event.keysym.lower()
		if key == "a":
			self.rotation = (self.rotation - 1) % 4
			self._update_rotation_label()
		elif key == "d":
			self.rotation = (self.rotation + 1) % 4
			self._update_rotation_label()
		elif key in ("q", "escape"):
			self.close()

	def _update_rotation_label(self):
		self.rotation_label.configure(text=f"Rotation: {self.rotation * 90}°")

	def _rotate(self, frame):
		rotations = {
			1: cv2.ROTATE_90_COUNTERCLOCKWISE,
			2: cv2.ROTATE_180,
			3: cv2.ROTATE_90_CLOCKWISE,
		}
		if self.rotation == 0:
			return frame
		return cv2.rotate(frame, rotations[self.rotation])

	def _current_source_frame(self):
		selected = self.notebook.tab(self.notebook.select(), "text")
		if selected == "Webcam":
			if self.camera is None:
				return None, "Webcam"
			success, frame = self.camera.read()
			return (frame, "Webcam") if success else (None, "Webcam")

		if self.file_kind == "video" and self.file_capture is not None and self.video_playing:
			success, frame = self.file_capture.read()
			if success:
				self.file_frame = frame
			else:
				self.video_playing = False
				self.play_button.configure(text="Phát")
				self.file_status.configure(text=f"{Path(self.file_path).name} · hết video")
		return self.file_frame, "File"

	def _process_frame(self, frame, source_name):
		frame = self._rotate(frame)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detect_faces(frame, gray, self.face_cascade, self.yunet_detector)
		display = frame.copy()
		for x, y, width, height in faces:
			cv2.rectangle(display, (x, y), (x + width, y + height), (60, 220, 60), 2)
			cv2.putText(
				display,
				"Face",
				(x, max(22, y - 8)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				(60, 220, 60),
				2,
				cv2.LINE_AA,
			)
		if self.yolo_model is not None:
			results = self.yolo_model.predict(
				source=frame,
				device=self.yolo_device,
				conf=self.args.confidence,
				verbose=False,
			)
			draw_yolo_detections(display, results[0])

		now = time.perf_counter()
		elapsed = now - self.last_frame_time
		self.last_frame_time = now
		if elapsed > 0:
			current_fps = 1 / elapsed
			self.fps = current_fps if self.fps == 0 else 0.85 * self.fps + 0.15 * current_fps

		height, width = display.shape[:2]
		cv2.rectangle(display, (0, 0), (width, min(76, height)), (25, 25, 25), -1)
		cv2.putText(
			display,
			f"{source_name}  |  {width}x{height}  |  FPS {self.fps:.1f}  |  Faces {len(faces)}",
			(12, 28),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.62,
			(245, 245, 245),
			1,
			cv2.LINE_AA,
		)
		model_name = self.model_var.get()
		cv2.putText(
			display,
			f"YOLO {model_name}  |  Device {self.yolo_device}  |  Confidence {self.args.confidence:.2f}",
			(12, 58),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(245, 245, 245),
			1,
			cv2.LINE_AA,
		)
		return display

	def _render(self, frame, view):
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(rgb)
		available_width = max(view.winfo_width() - 16, 320)
		available_height = max(view.winfo_height() - 16, 240)
		image.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)
		self.photo = ImageTk.PhotoImage(image=image)
		view.configure(image=self.photo, text="")

	def _update_frame(self):
		selected = self.notebook.tab(self.notebook.select(), "text")
		frame, source_name = self._current_source_frame()
		view = self.webcam_view if selected == "Webcam" else self.file_view
		if frame is not None:
			try:
				processed = self._process_frame(frame, source_name)
				self._render(processed, view)
			except Exception as error:
				view.configure(text=f"Lỗi xử lý ảnh: {error}", image="")
		elif selected == "Webcam" and self.camera is None:
			view.configure(text="Không thể mở webcam", image="")
		elif selected == "File" and self.file_path is None:
			view.configure(text="Chọn một ảnh hoặc video để bắt đầu", image="")

		delay = 20
		if selected == "File" and self.file_kind == "video" and self.video_playing:
			delay = max(5, int(1000 / self.video_fps))
		self.root.after(delay, self._update_frame)

	def close(self):
		self._stop_webcam()
		self._stop_file_video()
		self.root.destroy()


def main():
	args = parse_args()
	if not 0 <= args.confidence <= 1:
		raise SystemExit("--confidence must be between 0 and 1.")

	root = tk.Tk()
	try:
		WebcamFileApp(root, args)
	except Exception as error:
		messagebox.showerror("Không thể khởi động ứng dụng", str(error), parent=root)
		root.destroy()
		return
	root.mainloop()


if __name__ == "__main__":
	main()
