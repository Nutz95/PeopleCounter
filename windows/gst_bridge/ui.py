from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .service import GstBridgeService


class GstBridgeApp:
    def __init__(self, service: GstBridgeService, auto_close_seconds: float | None = None) -> None:
        self._service = service
        self._auto_close_seconds = auto_close_seconds
        self._root = tk.Tk()
        self._root.title("PeopleCounter GStreamer Bridge")
        self._root.geometry("1180x780")
        self._root.configure(bg="#10131a")
        self._cards: dict[str, tk.Frame] = {}
        self._labels: dict[str, tk.Label] = {}
        self._buttons: dict[str, ttk.Button] = {}
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
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

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

        right = tk.Frame(body, bg="#10131a", width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        right_tabs = ttk.Notebook(right)
        right_tabs.pack(fill=tk.BOTH, expand=True)

        metrics_tab = ttk.Frame(right_tabs, style="Dark.TFrame", padding=8)
        logs_tab = ttk.Frame(right_tabs, style="Dark.TFrame", padding=8)
        right_tabs.add(metrics_tab, text="Metrics")
        right_tabs.add(logs_tab, text="Execution log")

        canvas = tk.Canvas(left, bg="#10131a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=canvas.yview)
        self._items_frame = ttk.Frame(canvas, style="Dark.TFrame")
        self._items_frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._items_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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
        for child in self._items_frame.winfo_children():
            child.destroy()
        self._cards.clear()
        self._labels.clear()
        self._buttons.clear()
        items = self._service.list_items()
        for item in items:
            self._add_item_card(item.source_id, item.name, item.details)
        self._update_active_card()

    def _add_item_card(self, source_id: str, name: str, details: str) -> None:
        card = tk.Frame(self._items_frame, bg="#161b25", highlightbackground="#263246", highlightthickness=1, bd=0, padx=14, pady=14)
        card.pack(fill=tk.X, anchor=tk.W, pady=(0, 12))
        self._cards[source_id] = card

        title = tk.Label(card, text=name, bg="#161b25", fg="#f2f4f8", font=("Segoe UI", 12, "bold"), anchor="w")
        title.pack(anchor=tk.W)
        subtitle = tk.Label(card, text=details, bg="#161b25", fg="#95a0b5", font=("Segoe UI", 10), anchor="w", justify=tk.LEFT, wraplength=700)
        subtitle.pack(anchor=tk.W, pady=(4, 8))
        button = ttk.Button(card, text="Play now", command=lambda item_id=source_id: self._play_source(item_id), style="Accent.TButton")
        button.pack(anchor=tk.W)
        self._labels[source_id] = title
        self._buttons[source_id] = button

    def _play_source(self, source_id: str) -> None:
        try:
            self._service.select_source(source_id)
        except Exception:
            self._status_var.set(self._service.get_status_text())
        self._update_active_card()

    def _schedule_refresh(self) -> None:
        self._status_var.set(self._service.get_status_text())
        self._metrics_var.set("\n".join(self._service.get_metrics_snapshot().as_lines()))
        self._refresh_execution_log()
        self._update_active_card()
        if self._auto_close_seconds and self._auto_close_seconds > 0:
            self._root.after(int(self._auto_close_seconds * 1000), self._on_close)
            self._auto_close_seconds = None
        self._root.after(500, self._schedule_refresh)

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
            background = "#1b2431" if is_active else "#161b25"
            card.configure(bg=background, highlightbackground="#f59e0b" if is_active else "#263246", highlightthickness=2 if is_active else 1)
            self._labels[source_id].configure(bg=background, fg="#fff4d6" if is_active else "#f2f4f8")
            self._buttons[source_id].configure(text="Playing" if is_active else "Play now", state=tk.DISABLED if is_active else tk.NORMAL)

    def _on_close(self) -> None:
        self._service.stop()
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()
