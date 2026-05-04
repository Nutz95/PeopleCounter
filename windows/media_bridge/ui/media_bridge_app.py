from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ..models.media_item import MediaItem
from ..service.media_bridge_service import MediaBridgeService

THUMBNAIL_SIZE = (240, 135)


class MediaBridgeApp:
    def __init__(self, service: MediaBridgeService) -> None:
        self.service = service
        self.root = tk.Tk()
        self.root.title("PeopleCounter Static Media Bridge")
        self.root.geometry("1280x820")
        self.root.configure(bg="#10131a")
        self._thumbnail_refs: list[ImageTk.PhotoImage] = []
        self._item_cards: dict[str, tk.Frame] = {}
        self._item_names: dict[str, tk.Label] = {}
        self._item_details: dict[str, tk.Label] = {}
        self._item_badges: dict[str, tk.Label] = {}
        self._play_buttons: dict[str, ttk.Button] = {}
        self._current_source_var = tk.StringVar(value="Current source: none")
        self._stream_url_var = tk.StringVar(value=self.service.get_stream_url())
        self._build_layout()
        self.refresh_items()
        self._schedule_status_refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#10131a")
        style.configure("Dark.TLabel", background="#10131a", foreground="#f2f4f8")
        style.configure("Muted.TLabel", background="#10131a", foreground="#95a0b5")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        container = ttk.Frame(self.root, style="Dark.TFrame", padding=18)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container, style="Dark.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text="Static Media Bridge 4K30", style="Dark.TLabel", font=("Segoe UI", 20, "bold")).pack(anchor=tk.W)
        ttk.Label(header, textvariable=self._stream_url_var, style="Muted.TLabel", font=("Consolas", 10)).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(header, textvariable=self._current_source_var, style="Muted.TLabel", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(2, 12))

        actions = ttk.Frame(container, style="Dark.TFrame")
        actions.pack(fill=tk.X, pady=(0, 12))
        ttk.Button(actions, text="Refresh catalog", command=self.refresh_items, style="Accent.TButton").pack(side=tk.LEFT)

        canvas = tk.Canvas(container, bg="#10131a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        self._items_frame = ttk.Frame(canvas, style="Dark.TFrame")
        self._items_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._items_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def refresh_items(self) -> None:
        for child in self._items_frame.winfo_children():
            child.destroy()
        self._thumbnail_refs.clear()
        self._item_cards.clear()
        self._item_names.clear()
        self._item_details.clear()
        self._item_badges.clear()
        self._play_buttons.clear()
        items = self.service.refresh_items()
        if not items:
            ttk.Label(self._items_frame, text="No media or camera found.", style="Muted.TLabel").pack(anchor=tk.W)
            return
        for item in items:
            self._add_item_card(item)
        self._update_active_source_indicator()

    def _add_item_card(self, item: MediaItem) -> None:
        card = tk.Frame(
            self._items_frame,
            bg="#161b25",
            highlightbackground="#263246",
            highlightthickness=1,
            bd=0,
            padx=14,
            pady=14,
        )
        card.pack(fill=tk.X, anchor=tk.W, pady=(0, 14))
        self._item_cards[item.source_id] = card

        header = tk.Frame(card, bg="#161b25")
        header.pack(fill=tk.X)
        badge = tk.Label(
            header,
            text="READY",
            bg="#2b3444",
            fg="#d8deea",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
        )
        badge.pack(side=tk.RIGHT)
        self._item_badges[item.source_id] = badge

        body = tk.Frame(card, bg="#161b25")
        body.pack(fill=tk.X, expand=True)

        thumbnail = self._make_thumbnail(item)
        self._thumbnail_refs.append(thumbnail)
        preview = tk.Label(body, image=thumbnail, bg="#161b25", bd=0)
        preview.pack(side=tk.LEFT)

        info = tk.Frame(body, bg="#161b25")
        info.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(16, 0))
        name_label = tk.Label(info, text=item.name, bg="#161b25", fg="#f2f4f8", font=("Segoe UI", 12, "bold"))
        name_label.pack(anchor=tk.W)
        detail_label = tk.Label(
            info,
            text=item.details,
            bg="#161b25",
            fg="#95a0b5",
            justify=tk.LEFT,
            wraplength=760,
            font=("Segoe UI", 10),
        )
        detail_label.pack(anchor=tk.W, pady=(4, 10))
        self._item_names[item.source_id] = name_label
        self._item_details[item.source_id] = detail_label
        button = ttk.Button(
            info,
            text="Play now",
            command=lambda source_id=item.source_id: self._play_source(source_id),
            style="Accent.TButton",
        )
        button.pack(anchor=tk.W)
        self._play_buttons[item.source_id] = button

    def _make_thumbnail(self, item: MediaItem) -> ImageTk.PhotoImage:
        frame = None
        source_path = item.thumbnail_path
        if source_path is not None and source_path.exists():
            frame = self._read_preview_frame(source_path)
        if frame is None:
            frame = self._build_placeholder(item)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail(THUMBNAIL_SIZE)
        background = Image.new("RGB", THUMBNAIL_SIZE, color="#1c2230")
        x = (THUMBNAIL_SIZE[0] - image.width) // 2
        y = (THUMBNAIL_SIZE[1] - image.height) // 2
        background.paste(image, (x, y))
        return ImageTk.PhotoImage(background)

    def _read_preview_frame(self, path: Path) -> np.ndarray | None:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return None
        return frame

    def _build_placeholder(self, item: MediaItem) -> np.ndarray:
        image = Image.new("RGB", THUMBNAIL_SIZE, color="#1c2230")
        drawer = ImageDraw.Draw(image)
        drawer.rectangle((8, 8, THUMBNAIL_SIZE[0] - 8, THUMBNAIL_SIZE[1] - 8), outline="#3f4f72", width=2)
        drawer.text((16, 44), item.kind.upper(), fill="#f2f4f8")
        drawer.text((16, 72), item.name[:22], fill="#95a0b5")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def _play_source(self, source_id: str) -> None:
        try:
            self._current_source_var.set(self.service.get_status_text())
            self.root.update_idletasks()
            self.service.select_source(source_id)
            self._current_source_var.set(self.service.get_status_text())
            self._update_active_source_indicator()
        except Exception as exc:
            messagebox.showerror("Media Bridge", str(exc))

    def _schedule_status_refresh(self) -> None:
        self._current_source_var.set(self.service.get_status_text())
        self._update_active_source_indicator()
        self.root.after(1000, self._schedule_status_refresh)

    def _update_active_source_indicator(self) -> None:
        active_source_id = self.service.get_current_source_id()
        for source_id, card in self._item_cards.items():
            is_active = source_id == active_source_id
            background = "#1b2431" if is_active else "#161b25"
            card.configure(
                bg=background,
                highlightbackground="#f59e0b" if is_active else "#263246",
                highlightthickness=2 if is_active else 1,
            )
            if source_id in self._item_names:
                self._item_names[source_id].configure(bg=background, fg="#fff4d6" if is_active else "#f2f4f8")
            if source_id in self._item_details:
                self._item_details[source_id].configure(bg=background, fg="#ffd58a" if is_active else "#95a0b5")
            if source_id in self._item_badges:
                self._item_badges[source_id].configure(
                    text="ACTIVE" if is_active else "READY",
                    bg="#f59e0b" if is_active else "#2b3444",
                    fg="#1a1304" if is_active else "#d8deea",
                )
            if source_id in self._play_buttons:
                self._play_buttons[source_id].configure(
                    text="Playing" if is_active else "Play now",
                    state=tk.DISABLED if is_active else tk.NORMAL,
                )

    def _on_close(self) -> None:
        self.service.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()