"""Select a vehicle plate and export its segmented characters as PNG files."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


def _character_boxes(mask):
	height, width = mask.shape
	minimum_height = max(8, int(height * 0.25))
	maximum_height = int(height * 0.98)
	minimum_width = max(3, int(width * 0.008))
	maximum_width = int(width * 0.25)
	minimum_area = max(20, int(width * height * 0.0005))
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	boxes = []
	for contour in contours:
		x, y, box_width, box_height = cv2.boundingRect(contour)
		area = cv2.contourArea(contour)
		if not minimum_height <= box_height <= maximum_height:
			continue
		if not minimum_width <= box_width <= maximum_width or area < minimum_area:
			continue
		aspect_ratio = box_width / box_height
		if not 0.06 <= aspect_ratio <= 1.15:
			continue
		boxes.append((x, y, box_width, box_height))
	return boxes


def _order_boxes_by_rows(boxes):
	if not boxes:
		return []

	median_height = float(np.median([box[3] for box in boxes]))
	rows = []
	for box in sorted(boxes, key=lambda item: item[1] + item[3] / 2):
		center_y = box[1] + box[3] / 2
		for row in rows:
			row_center = float(np.mean([item[1] + item[3] / 2 for item in row]))
			if abs(center_y - row_center) <= median_height * 0.45:
				row.append(box)
				break
		else:
			rows.append([box])

	ordered = []
	for row in rows:
		ordered.extend(sorted(row, key=lambda item: item[0]))
	return ordered


def segment_characters(plate_image):
	"""Return tightly cropped binary characters found in a plate image."""
	if plate_image is None or plate_image.size == 0:
		return []

	gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	_, normal_mask = cv2.threshold(
		gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
	)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	candidates = []
	for mask in (normal_mask, cv2.bitwise_not(normal_mask)):
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		boxes = _character_boxes(mask)
		if boxes:
			candidates.append((abs(len(boxes) - 7), -len(boxes), mask, boxes))

	if not candidates:
		return []

	_, _, character_mask, boxes = min(candidates, key=lambda item: (item[0], item[1]))
	ordered_boxes = _order_boxes_by_rows(boxes)
	padding = max(2, int(min(gray.shape) * 0.025))
	characters = []
	for x, y, width, height in ordered_boxes:
		x1 = max(0, x - padding)
		y1 = max(0, y - padding)
		x2 = min(character_mask.shape[1], x + width + padding)
		y2 = min(character_mask.shape[0], y + height + padding)
		character = character_mask[y1:y2, x1:x2]
		characters.append(cv2.copyMakeBorder(
			character, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
		))
	return characters


class PlateCharacterApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Tách ký tự biển số xe")
		self.root.geometry("1100x720")
		self.root.minsize(820, 560)
		self.root.configure(bg="#171b1f")

		self.source_path = None
		self.source_image = None
		self.plate_image = None
		self.character_images = []
		self.preview_photo = None
		self.character_photos = []

		self._build_interface()
		self.root.bind_all("<Control-s>", self._on_save_shortcut)

	def _build_interface(self):
		style = ttk.Style(self.root)
		style.theme_use("clam")
		style.configure("TFrame", background="#171b1f")
		style.configure("TLabel", background="#171b1f", foreground="#e9eef2")
		style.configure("TButton", padding=(10, 7))

		header = ttk.Frame(self.root, padding=(18, 14))
		header.pack(fill="x")
		ttk.Label(
			header, text="PLATE CHARACTER EXTRACTOR", font=("Segoe UI Semibold", 15)
		).pack(side="left")
		self.status = ttk.Label(header, text="Chưa chọn ảnh")
		self.status.pack(side="right")

		controls = ttk.Frame(self.root, padding=(18, 0, 18, 12))
		controls.pack(fill="x")
		self.open_button = ttk.Button(
			controls, text="Mở ảnh", command=self._open_image
		)
		self.open_button.pack(side="left")
		self.select_button = ttk.Button(
			controls,
			text="Khoanh vùng biển số",
			command=self._select_plate_region,
			state="disabled",
		)
		self.select_button.pack(side="left", padx=(8, 0))
		self.full_image_button = ttk.Button(
			controls,
			text="Dùng toàn ảnh",
			command=self._use_full_image,
			state="disabled",
		)
		self.full_image_button.pack(side="left", padx=(8, 0))
		self.segment_button = ttk.Button(
			controls,
			text="Tách ký tự",
			command=self._segment,
			state="disabled",
		)
		self.segment_button.pack(side="left", padx=(8, 0))
		self.save_button = ttk.Button(
			controls,
			text="Lưu PNG",
			command=self._save_characters,
			state="disabled",
		)
		self.save_button.pack(side="left", padx=(8, 0))

		content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
		content.pack(fill="both", expand=True, padx=18, pady=(0, 18))
		left = ttk.Frame(content, padding=10)
		right = ttk.Frame(content, padding=10)
		content.add(left, weight=3)
		content.add(right, weight=2)

		ttk.Label(left, text="Vùng biển số", font=("Segoe UI Semibold", 11)).pack(
			anchor="w", pady=(0, 8)
		)
		self.image_view = tk.Label(
			left,
			text="Mở ảnh xe hoặc ảnh biển số để bắt đầu",
			bg="#090c0f",
			fg="#bac5cc",
			font=("Segoe UI", 12),
		)
		self.image_view.pack(fill="both", expand=True)

		self.character_heading = ttk.Label(
			right, text="Ký tự tách được: 0", font=("Segoe UI Semibold", 11)
		)
		self.character_heading.pack(anchor="w", pady=(0, 8))
		self.character_gallery = ttk.Frame(right)
		self.character_gallery.pack(fill="both", expand=True)

	def _open_image(self):
		path = filedialog.askopenfilename(
			parent=self.root,
			title="Chọn ảnh xe hoặc ảnh biển số",
			filetypes=(
				("Tệp ảnh", "*.bmp *.jpeg *.jpg *.png *.tif *.tiff *.webp"),
				("Tất cả tệp", "*.*"),
			),
		)
		if not path:
			return
		try:
			encoded = np.fromfile(path, dtype=np.uint8)
			image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
		except (OSError, cv2.error) as error:
			messagebox.showerror("Không đọc được ảnh", str(error), parent=self.root)
			return
		if image is None:
			messagebox.showerror(
				"Không đọc được ảnh", "Định dạng ảnh không được hỗ trợ.", parent=self.root
			)
			return

		self.source_path = Path(path)
		self.source_image = image
		self.plate_image = image.copy()
		self.character_images = []
		self.select_button.configure(state="normal")
		self.full_image_button.configure(state="normal")
		self.segment_button.configure(state="normal")
		self.save_button.configure(state="disabled")
		self.status.configure(text=self.source_path.name)
		self.character_heading.configure(text="Ký tự tách được: 0")
		self._clear_character_gallery()
		self._show_plate_image()

	def _select_plate_region(self):
		if self.source_image is None:
			return
		image_height, image_width = self.source_image.shape[:2]
		scale = min(1.0, 1400 / image_width, 850 / image_height)
		if scale < 1:
			preview = cv2.resize(
				self.source_image,
				(int(image_width * scale), int(image_height * scale)),
				interpolation=cv2.INTER_AREA,
			)
		else:
			preview = self.source_image
		try:
			x, y, width, height = cv2.selectROI(
				"Select plate area - drag and press Enter",
				preview,
				showCrosshair=True,
				fromCenter=False,
			)
			cv2.destroyWindow("Select plate area - drag and press Enter")
		except cv2.error as error:
			messagebox.showerror("Không thể chọn vùng", str(error), parent=self.root)
			return
		if width == 0 or height == 0:
			return

		x1 = max(0, int(x / scale))
		y1 = max(0, int(y / scale))
		x2 = min(image_width, int((x + width) / scale))
		y2 = min(image_height, int((y + height) / scale))
		self.plate_image = self.source_image[y1:y2, x1:x2].copy()
		self.character_images = []
		self.character_heading.configure(text="Ký tự tách được: 0")
		self.save_button.configure(state="disabled")
		self._clear_character_gallery()
		self.status.configure(text=f"Đã chọn vùng {x2 - x1} x {y2 - y1}")
		self._show_plate_image()

	def _use_full_image(self):
		if self.source_image is None:
			return
		self.plate_image = self.source_image.copy()
		self.character_images = []
		self.character_heading.configure(text="Ký tự tách được: 0")
		self.save_button.configure(state="disabled")
		self._clear_character_gallery()
		self.status.configure(text="Đang dùng toàn ảnh làm vùng biển số")
		self._show_plate_image()

	def _show_plate_image(self):
		if self.plate_image is None:
			return
		rgb = cv2.cvtColor(self.plate_image, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(rgb)
		image.thumbnail((720, 500), Image.Resampling.LANCZOS)
		self.preview_photo = ImageTk.PhotoImage(image=image)
		self.image_view.configure(image=self.preview_photo, text="")

	def _clear_character_gallery(self):
		for child in self.character_gallery.winfo_children():
			child.destroy()
		self.character_photos = []

	def _segment(self):
		if self.plate_image is None:
			return
		self.character_images = segment_characters(self.plate_image)
		self._clear_character_gallery()
		if not self.character_images:
			self.character_heading.configure(text="Ký tự tách được: 0")
			self.save_button.configure(state="disabled")
			self.status.configure(text="Không tìm thấy ký tự; hãy kiểm tra vùng biển số")
			return

		self.character_heading.configure(
			text=f"Ký tự tách được: {len(self.character_images)}"
		)
		self.save_button.configure(state="normal")
		self.status.configure(text="Tách ký tự hoàn tất")
		for index, character in enumerate(self.character_images, start=1):
			tile = ttk.Frame(self.character_gallery, padding=5)
			tile.grid(row=(index - 1) // 4, column=(index - 1) % 4, padx=3, pady=4)
			preview = Image.fromarray(character)
			preview.thumbnail((64, 82), Image.Resampling.NEAREST)
			photo = ImageTk.PhotoImage(image=preview)
			self.character_photos.append(photo)
			tk.Label(tile, image=photo).pack()
			ttk.Label(tile, text=f"{index:02d}").pack(pady=(3, 0))

	def _save_characters(self):
		if not self.character_images or self.source_path is None:
			return
		directory = filedialog.askdirectory(
			parent=self.root,
			title="Chọn thư mục lưu ảnh ký tự PNG",
		)
		if not directory:
			return
		output_directory = Path(directory) / f"{self.source_path.stem}_characters"
		try:
			output_directory.mkdir(parents=True, exist_ok=True)
			for index, character in enumerate(self.character_images, start=1):
				output_path = output_directory / f"ky_tu_{index:02d}.png"
				success, encoded = cv2.imencode(".png", character)
				if not success:
					raise OSError(f"Không mã hóa được {output_path.name}")
				encoded.tofile(str(output_path))
		except OSError as error:
			messagebox.showerror("Không lưu được ảnh", str(error), parent=self.root)
			return
		self.status.configure(text=f"Đã lưu {len(self.character_images)} ảnh PNG")
		messagebox.showinfo(
			"Lưu hoàn tất",
			f"Đã lưu {len(self.character_images)} ảnh ký tự vào:\n{output_directory}",
			parent=self.root,
		)

	def _on_save_shortcut(self, _event=None):
		if not self.character_images:
			messagebox.showinfo(
				"Chưa có ký tự",
				"Hãy tách ký tự trước khi lưu bằng Ctrl+S.",
				parent=self.root,
			)
			return "break"
		self._save_characters()
		return "break"


def main():
	root = tk.Tk()
	PlateCharacterApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
