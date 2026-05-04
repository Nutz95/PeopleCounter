from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from .models import MediaItem
from .service import GstBridgeService
from .thumbnail_loader import THUMBNAIL_SIZE, load_thumbnail_image, thumbnails_enabled

try:
    from PIL import ImageTk
except ImportError:  # pragma: no cover - optional dependency already declared in runtime requirements
    ImageTk = None


class GstBridgeApp:
    def __init__(self, service: GstBridgeService, auto_close_seconds: float | None = None) -> None:
        self._service = service
        self._auto_close_seconds = auto_close_seconds
        self._root = tk.Tk()
        self._root.title("PeopleCounter GStreamer Bridge")
        self._root.geometry("1440x900")
        self._root.configure(bg="#10131a")
        self._cards: dict[str, tk.Frame] = {}
        self._labels: dict[str, tk.Label] = {}
        self._detail_labels: dict[str, tk.Label] = {}
        self._thumb_labels: dict[str, tk.Label] = {}
        self._thumbnail_images: dict[str, object] = {}
        self._items_by_kind: dict[str, list[MediaItem]] = {"camera": [], "image": [], "video": []}
        self._tab_frames: dict[str, ttk.Frame] = {}
        self._tab_canvases: dict[str, tk.Canvas] = {}
        self._tab_counts: dict[str, int] = {}
        self._active_scroll_canvas: tk.Canvas | None = None
        self._metadata_text: tk.Text | None = None
        self._last_metadata_signature = ""
        self._log_text: tk.Text | None = None
        self._last_log_signature = ""
        self._status_var = tk.StringVar(value="starting…")
        self._stream_var = tk.StringVar(value=service.get_stream_url())
        self._metrics_var = tk.StringVar(value="metrics pending…")
        self._build_layout()
        self.refresh_items()
        self._schedule_refresh()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#10131a")
        style.configure("Dark.TLabel", background="#10131a", foreground="#f2f4f8")
        style.configure("Muted.TLabel", background="#10131a", foreground="#95a0b5")

        container = ttk.Frame(self._root, style="Dark.TFrame", padding=18)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container, style="Dark.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text="GStreamer + MediaMTX Bridge", style="Dark.TLabel", font=("Segoe UI", 20, "bold")).pack(anchor=tk.W)
        ttk.Label(header, textvariable=self._stream_var, style="Muted.TLabel", font=("Consolas", 10)).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(header, textvariable=self._status_var, style="Muted.TLabel", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(2, 12))

        body = ttk.Frame(container, style="Dark.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg="#10131a")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(body, bg="#10131a", width=420)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        right_tabs = ttk.Notebook(right)
        right_tabs.pack(fill=tk.BOTH, expand=True)

        metadata_tab = ttk.Frame(right_tabs, style="Dark.TFrame", padding=8)
        metrics_tab = ttk.Frame(right_tabs, style="Dark.TFrame", padding=8)
        logs_tab = ttk.Frame(right_tabs, style="Dark.TFrame", padding=8)
        right_tabs.add(metadata_tab, text="Metadata")
        right_tabs.add(metrics_tab, text="Metrics")
        right_tabs.add(logs_tab, text="Execution log")

        media_tabs = ttk.Notebook(left)
        media_tabs.pack(fill=tk.BOTH, expand=True)
        for kind, label in (("camera", "Cameras"), ("image", "Images"), ("video", "Videos")):
            tab = ttk.Frame(media_tabs, style="Dark.TFrame", padding=6)
            media_tabs.add(tab, text=label)
            canvas = tk.Canvas(tab, bg="#10131a", highlightthickness=0)
            scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
            frame = ttk.Frame(canvas, style="Dark.TFrame")
            frame.bind("<Configure>", lambda _event, canvas_ref=canvas: canvas_ref.configure(scrollregion=canvas_ref.bbox("all")))
            canvas.create_window((0, 0), window=frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.bind("<Enter>", lambda _event, canvas_ref=canvas: self._activate_scroll_canvas(canvas_ref))
            canvas.bind("<Leave>", lambda _event: self._deactivate_scroll_canvas())
            self._tab_frames[kind] = frame
            self._tab_canvases[kind] = canvas
            self._tab_counts[kind] = 0

        self._root.bind_all("<MouseWheel>", self._on_mousewheel)
        self._root.bind_all("<Button-4>", self._on_mousewheel)
        self._root.bind_all("<Button-5>", self._on_mousewheel)

        tk.Label(metadata_tab, text="Current source metadata", bg="#10131a", fg="#f2f4f8", font=("Segoe UI", 13, "bold"), anchor="w").pack(fill=tk.X, pady=(0, 8))
        metadata_frame = tk.Frame(metadata_tab, bg="#161b25", highlightbackground="#263246", highlightthickness=1, bd=0)
        metadata_frame.pack(fill=tk.BOTH, expand=True)
        metadata_scrollbar = ttk.Scrollbar(metadata_frame, orient=tk.VERTICAL)
        metadata_text = tk.Text(
            metadata_frame,
            bg="#0f141d",
            fg="#d8e1ff",
            insertbackground="#d8e1ff",
            relief=tk.FLAT,
            wrap=tk.WORD,
            font=("Consolas", 10),
            padx=12,
            pady=12,
            yscrollcommand=metadata_scrollbar.set,
        )
        metadata_scrollbar.config(command=metadata_text.yview)
        metadata_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        metadata_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        metadata_text.configure(state=tk.DISABLED)
        self._metadata_text = metadata_text

        tk.Label(metrics_tab, text="Runtime metrics", bg="#10131a", fg="#f2f4f8", font=("Segoe UI", 13, "bold"), anchor="w").pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            metrics_tab,
            textvariable=self._metrics_var,
            bg="#161b25",
            fg="#f2f4f8",
            justify=tk.LEFT,
            anchor="nw",
            padx=12,
            pady=12,
            font=("Consolas", 9),
            wraplength=320,
        ).pack(fill=tk.BOTH, expand=True)

        tk.Label(logs_tab, text="Play trace + GStreamer logs", bg="#10131a", fg="#f2f4f8", font=("Segoe UI", 13, "bold"), anchor="w").pack(fill=tk.X, pady=(0, 8))
        log_frame = tk.Frame(logs_tab, bg="#161b25", highlightbackground="#263246", highlightthickness=1, bd=0)
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL)
        log_text = tk.Text(
            log_frame,
            bg="#0f141d",
            fg="#d8e1ff",
            insertbackground="#d8e1ff",
            relief=tk.FLAT,
            wrap=tk.NONE,
            font=("Consolas", 9),
            padx=10,
            pady=10,
            yscrollcommand=log_scrollbar.set,
        )
        log_scrollbar.config(command=log_text.yview)
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        log_text.configure(state=tk.DISABLED)
        self._log_text = log_text

    def refresh_items(self) -> None:
        for frame in self._tab_frames.values():
            for child in frame.winfo_children():
                child.destroy()
        self._cards.clear()
        self._labels.clear()
        self._detail_labels.clear()
        self._thumb_labels.clear()
        self._thumbnail_images.clear()
        self._items_by_kind = {"camera": [], "image": [], "video": []}
        self._tab_counts = {"camera": 0, "image": 0, "video": 0}
        items = self._service.list_items()
        for item in items:
            self._items_by_kind.setdefault(item.kind, []).append(item)
        for kind in ("camera", "image", "video"):
            for item in self._items_by_kind.get(kind, []):
                self._add_item_card(kind, item)
        self._update_active_card()

    def _add_item_card(self, kind: str, item: MediaItem) -> None:
        source_id = item.source_id
        parent = self._tab_frames[kind]
        index = self._tab_counts[kind]
        row, column = divmod(index, 2)
        self._tab_counts[kind] += 1

        card = tk.Frame(parent, bg="#161b25", highlightbackground="#263246", highlightthickness=2, bd=0, padx=12, pady=12, cursor="hand2")
        card.grid(row=row, column=column, sticky="nwe", padx=10, pady=10)
        parent.grid_columnconfigure(column, weight=1)
        self._cards[source_id] = card

        thumb_container = tk.Frame(
            card,
            bg="#0f141d",
            width=THUMBNAIL_SIZE[0],
            height=THUMBNAIL_SIZE[1],
            highlightbackground="#263246",
            highlightthickness=1,
            bd=0,
            cursor="hand2",
        )
        thumb_container.pack(fill=tk.X)
        thumb_container.pack_propagate(False)

        thumb_label = tk.Label(
            thumb_container,
            text="CAM" if item.kind == "camera" else "Preview",
            bg="#0f141d",
            fg="#95a0b5",
            compound=tk.CENTER,
            anchor="center",
            cursor="hand2",
        )
        thumb_label.pack(fill=tk.BOTH, expand=True)
        self._thumb_labels[source_id] = thumb_label

        title = tk.Label(card, text=item.name, bg="#161b25", fg="#f2f4f8", font=("Segoe UI", 12, "bold"), anchor="w", justify=tk.LEFT, wraplength=THUMBNAIL_SIZE[0])
        title.pack(anchor=tk.W, fill=tk.X, pady=(10, 0))
        subtitle = tk.Label(card, text=item.details, bg="#161b25", fg="#95a0b5", font=("Segoe UI", 9), anchor="w", justify=tk.LEFT, wraplength=THUMBNAIL_SIZE[0])
        subtitle.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
        self._labels[source_id] = title
        self._detail_labels[source_id] = subtitle
        self._bind_card_click(card, source_id)
        if item.kind in {"video", "image"} and thumbnails_enabled() and ImageTk is not None:
            threading.Thread(target=self._load_thumbnail, args=(source_id, item), daemon=True, name=f"thumb:{item.name}").start()

    def _bind_card_click(self, card: tk.Widget, source_id: str) -> None:
        card.bind("<Button-1>", lambda _event, item_id=source_id: self._play_source(item_id))
        for child in card.winfo_children():
            child.bind("<Button-1>", lambda _event, item_id=source_id: self._play_source(item_id))
            child.configure(cursor="hand2")
            for grandchild in child.winfo_children():
                grandchild.bind("<Button-1>", lambda _event, item_id=source_id: self._play_source(item_id))
                grandchild.configure(cursor="hand2")

    def _activate_scroll_canvas(self, canvas: tk.Canvas) -> None:
        self._active_scroll_canvas = canvas

    def _deactivate_scroll_canvas(self) -> None:
        self._active_scroll_canvas = None

    def _on_mousewheel(self, event) -> None:
        canvas = self._active_scroll_canvas
        if canvas is None:
            return
        if hasattr(event, "delta") and event.delta:
            delta = -1 * int(event.delta / 120)
            if delta != 0:
                canvas.yview_scroll(delta, "units")
            return
        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")

    def _load_thumbnail(self, source_id: str, item: MediaItem) -> None:
        image = load_thumbnail_image(item, self._service.get_ffmpeg_path())
        if image is None:
            return
        self._root.after(0, lambda img=image, item_id=source_id: self._set_thumbnail(item_id, img))

    def _set_thumbnail(self, source_id: str, image) -> None:
        if ImageTk is None:
            return
        label = self._thumb_labels.get(source_id)
        if label is None or not label.winfo_exists():
            return
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, text="")
        self._thumbnail_images[source_id] = photo

    def _play_source(self, source_id: str) -> None:
        try:
            self._service.select_source(source_id)
        except Exception:
            self._status_var.set(self._service.get_status_text())
        self._update_active_card()

    def _schedule_refresh(self) -> None:
        self._status_var.set(self._service.get_status_text())
        self._metrics_var.set("\n".join(self._service.get_metrics_snapshot().as_lines()))
        self._refresh_metadata()
        self._refresh_execution_log()
        self._update_active_card()
        if self._auto_close_seconds and self._auto_close_seconds > 0:
            self._root.after(int(self._auto_close_seconds * 1000), self._on_close)
            self._auto_close_seconds = None
        self._root.after(500, self._schedule_refresh)

    def _refresh_metadata(self) -> None:
        if self._metadata_text is None:
            return
        rendered = "\n".join(self._service.get_current_source_metadata_lines())
        if rendered == self._last_metadata_signature:
            return
        self._last_metadata_signature = rendered
        self._metadata_text.configure(state=tk.NORMAL)
        self._metadata_text.delete("1.0", tk.END)
        self._metadata_text.insert(tk.END, rendered)
        if rendered:
            self._metadata_text.insert(tk.END, "\n")
        self._metadata_text.configure(state=tk.DISABLED)

    def _refresh_execution_log(self) -> None:
        if self._log_text is None:
            return
        lines = self._service.get_execution_log_lines()
        rendered = "\n".join(lines)
        if rendered == self._last_log_signature:
            return
        self._last_log_signature = rendered
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.insert(tk.END, rendered)
        if rendered:
            self._log_text.insert(tk.END, "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _update_active_card(self) -> None:
        active_id = self._service.get_current_source_id()
        for source_id, card in self._cards.items():
            is_active = source_id == active_id
            background = "#18261d" if is_active else "#161b25"
            border = "#22c55e" if is_active else "#263246"
            card.configure(bg=background, highlightbackground=border, highlightthickness=3 if is_active else 2)
            self._labels[source_id].configure(bg=background, fg="#fff4d6" if is_active else "#f2f4f8")
            self._detail_labels[source_id].configure(bg=background, fg="#ffe4b0" if is_active else "#95a0b5")
            thumb = self._thumb_labels[source_id]
            thumb.configure(bg="#102014" if is_active else "#0f141d")
            thumb.configure(highlightbackground=border, highlightthickness=2 if is_active else 1)

    def _on_close(self) -> None:
        self._root.unbind_all("<MouseWheel>")
        self._root.unbind_all("<Button-4>")
        self._root.unbind_all("<Button-5>")
        self._service.stop()
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()
