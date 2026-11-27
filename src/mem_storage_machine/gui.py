from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np

from . import hopfield as hopfield_model
from .workflows import run_hopfield, run_som


@dataclass
class App:
    hopfield: ttk.Frame
    som: ttk.Frame


def build_app(root, data_dir: Path) -> App:
    """在傳入的視窗建立 Hopfield 與 SOM 分頁。"""
    data_dir = Path(data_dir)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    hop = ttk.Frame(notebook, padding=10)
    som = ttk.Frame(notebook, padding=10)
    notebook.add(hop, text="Hopfield")
    notebook.add(som, text="SOM")

    # ---------- Hopfield ----------
    hop_folder = data_dir / "hopfield"
    hop_controls = ttk.Frame(hop)
    hop_controls.pack(fill="x")

    ttk.Label(hop_controls, text="資料集").pack(side="left")
    hop.dataset = ttk.Combobox(
        hop_controls, values=["Basic", "Bonus"], state="readonly", width=10
    )
    hop.dataset.current(0)
    hop.dataset.pack(side="left", padx=5)

    hop.canvas = tk.Canvas(hop, width=690, height=300, background="#eeeeee")
    hop.canvas.pack(fill="both", expand=True, pady=8)
    hop.status = ttk.Label(hop)
    hop.status.pack(anchor="w")

    hop.cases = []
    hop.case_index = 0
    noise_weights = None

    def draw_case(target, cue, recalled):
        canvas = hop.canvas
        canvas.delete("all")
        size = 20
        left = 10

        for tag, title, pattern in (
            ("target", "原圖", target),
            ("cue", "測試輸入", cue),
            ("recalled", "回想結果", recalled),
        ):
            height, width = pattern.shape
            canvas.create_text(left, 8, text=title, anchor="nw")

            for row in range(height):
                for col in range(width):
                    x = left + col * size
                    y = 32 + row * size
                    color = "black" if pattern[row, col] == 1 else "white"
                    canvas.create_rectangle(
                        x, y, x + size, y + size,
                        fill=color, outline="", tags=(tag,),
                    )

            left += width * size + 24

        hop.status.config(text=f"第 {hop.case_index + 1} / {len(hop.cases)} 組")

    def show_current_case():
        case = hop.cases[hop.case_index]
        draw_case(case.target, case.cue, case.recalled)

    def on_hop_run():
        nonlocal noise_weights
        name = hop.dataset.get()
        try:
            hop.cases = run_hopfield(
                hop_folder / f"{name}_Training.txt",
                hop_folder / f"{name}_Testing.txt",
            )
        except (OSError, ValueError) as error:
            hop.cases = []
            hop.canvas.delete("all")
            hop.status.config(text=str(error))
            return

        hop.case_index = 0
        noise_weights = None
        if hop.cases:
            show_current_case()
        else:
            hop.canvas.delete("all")
            hop.status.config(text="沒有可顯示的圖案。")

    def on_next_case():
        if hop.cases:
            hop.case_index = (hop.case_index + 1) % len(hop.cases)
            show_current_case()

    def on_noise():
        nonlocal noise_weights
        if not hop.cases:
            return

        if noise_weights is None:
            memories = np.stack([case.target.ravel() for case in hop.cases])
            noise_weights = hopfield_model.train(memories)

        target = hop.cases[hop.case_index].target
        cue = hopfield_model.add_noise(target.copy(), flips=1, seed=hop.case_index)
        recalled = hopfield_model.recall(noise_weights, cue.ravel().copy())
        draw_case(target, cue, recalled.reshape(target.shape))

    hop.run = ttk.Button(hop_controls, text="執行", command=on_hop_run)
    hop.next_case = ttk.Button(hop_controls, text="下一組", command=on_next_case)
    hop.noise = ttk.Button(hop_controls, text="加雜訊", command=on_noise)
    for button in (hop.run, hop.next_case, hop.noise):
        button.pack(side="left", padx=3)

    # ---------- SOM ----------
    som_folder = data_dir / "som"
    som_controls = ttk.Frame(som)
    som_controls.pack(fill="x")

    filenames = sorted(
        (p.name for p in som_folder.glob("*")
         if p.is_file() and p.suffix.lower() == ".txt"),
        key=str.casefold,
    )
    ttk.Label(som_controls, text="資料集").pack(side="left")
    som.dataset = ttk.Combobox(
        som_controls, values=filenames, state="readonly", width=18
    )
    if filenames:
        som.dataset.current(0)
    som.dataset.pack(side="left", padx=5)

    ttk.Label(som_controls, text="迭代次數").pack(side="left")
    som.iterations = ttk.Entry(som_controls, width=7)
    som.iterations.insert(0, "100")
    som.iterations.pack(side="left", padx=5)

    plot_width, plot_height = 660, 340
    som.canvas = tk.Canvas(
        som, width=plot_width, height=plot_height, background="white"
    )
    som.canvas.pack(fill="both", expand=True, pady=8)
    som.status = ttk.Label(som)
    som.status.pack(anchor="w")

    som.result = None
    som.frame_index = 0

    def draw_som_frame():
        result = som.result
        canvas = som.canvas
        canvas.delete("all")

        # 每一幀共用相同座標範圍，切換時才看得出權重移動。
        samples = result.features[:, :2]
        initial = result.history[0][:, :, :2].reshape(-1, 2)
        bounds = np.vstack((samples, initial))
        low = bounds.min(axis=0)
        span = bounds.max(axis=0) - low
        span[span == 0] = 1

        def project(points):
            unit = (points[..., :2] - low) / span
            x = 20 + unit[..., 0] * (plot_width - 40)
            y = plot_height - 20 - unit[..., 1] * (plot_height - 40)
            return np.stack((x, y), axis=-1)

        colors = ("#2563eb", "#ea580c", "#15803d", "#9333ea", "#be123c")
        label_colors = {
            label: colors[i % len(colors)]
            for i, label in enumerate(dict.fromkeys(result.labels))
        }

        for (x, y), label in zip(project(samples), result.labels):
            canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3,
                fill=label_colors[label], outline="", tags=("sample",),
            )

        nodes = project(result.history[som.frame_index])
        rows, cols = nodes.shape[:2]
        for row in range(rows):
            for col in range(cols):
                x, y = nodes[row, col]
                if col + 1 < cols:
                    nx, ny = nodes[row, col + 1]
                    canvas.create_line(x, y, nx, ny, fill="#475569", tags=("edge",))
                if row + 1 < rows:
                    nx, ny = nodes[row + 1, col]
                    canvas.create_line(x, y, nx, ny, fill="#475569", tags=("edge",))

        for x, y in nodes.reshape(-1, 2):
            canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill="white", outline="black", tags=("node",),
            )

        som.status.config(
            text=f"迭代 {som.frame_index} / {len(result.history) - 1}"
        )

    def on_som_run():
        try:
            iterations = int(som.iterations.get())
            if iterations < 1:
                raise ValueError
        except ValueError:
            som.status.config(text="[SOM_ITERATIONS] 迭代次數必須為正整數。")
            return

        name = som.dataset.get()
        if not name:
            som.status.config(text="請先選擇 SOM 資料檔。")
            return

        try:
            som.result = run_som(som_folder / name, iterations=iterations)
        except (OSError, ValueError) as error:
            som.status.config(text=str(error))
            return
        features = som.result.features
        history = som.result.history

        for frame in sorted({0, 1, 10, 100, 1000, len(history) - 1}):
            if frame >= len(history):
                continue

            nodes = history[frame].reshape(-1, features.shape[1])
            distances = np.linalg.norm(features[:, None, :] - nodes[None, :, :], axis=2)
            error = distances.min(axis=1).mean()
            print(f"第 {frame} 幀，量化誤差 = {error:.6f}")
        som.frame_index = 0
        draw_som_frame()

    def on_next_frame():
        if som.result is not None:
            som.frame_index = (som.frame_index + 1) % len(som.result.history)
            draw_som_frame()

    som.run = ttk.Button(som_controls, text="執行", command=on_som_run)
    som.next_frame = ttk.Button(som_controls, text="下一幀", command=on_next_frame)
    som.run.pack(side="left", padx=3)
    som.next_frame.pack(side="left", padx=3)
    ttk.Label(som_controls, text="跳到第幾幀").pack(side="left", padx=(10, 2))

    frame_entry = ttk.Entry(som_controls, width=6)
    frame_entry.pack(side="left")

    def jump_to_frame(event=None):
        if som.result is None:
            som.status.config(text="請先執行 SOM 訓練。")
            return

        last_frame = len(som.result.history) - 1
        try:
            index = int(frame_entry.get())
        except ValueError:
            som.status.config(text=f"請輸入 0 到 {last_frame} 之間的整數。")
            return

        if not 0 <= index <= last_frame:
            som.status.config(text=f"幀號超出範圍，可選 0 到 {last_frame}。")
            return

        som.frame_index = index
        draw_som_frame()

    ttk.Button(som_controls, text="跳轉", command=jump_to_frame).pack(
        side="left", padx=3
    )
    frame_entry.bind("<Return>", jump_to_frame)

    return App(hopfield=hop, som=som)
