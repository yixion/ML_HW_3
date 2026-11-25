"""對真正 Tk 元件輸入、點擊並讀回畫布；不進入 mainloop。"""
import numpy as np
import pytest

from mem_storage_machine.data import load_patterns
from mem_storage_machine.gui import build_app

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def tk_root(request):
    try:
        import tkinter as tk
    except ImportError:
        if request.config.getoption("--require-gui"):
            pytest.fail("Python 未安裝 Tkinter")
        pytest.skip("Python 未安裝 Tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        if request.config.getoption("--require-gui"):
            pytest.fail(f"Tk 無法啟動：{error}")
        pytest.skip(f"Tk 無法啟動：{error}")
    try:
        root.withdraw()
        yield root
    finally:
        root.destroy()


@pytest.fixture
def window(tk_root):
    from tkinter import Toplevel

    top = Toplevel(tk_root)
    top.withdraw()
    try:
        yield top
    finally:
        top.destroy()


def pixels(canvas, tag, shape):
    # row-major 順序；黑白由同一個畫布的兩個顏色判斷。
    items = canvas.find_withtag(tag)
    assert len(items) == np.prod(shape)
    return np.array([canvas.itemcget(item, "fill") for item in items]).reshape(shape)


def assert_image(colors, pattern, palette):
    expected = np.where(pattern == 1, palette[1], palette[-1])
    np.testing.assert_array_equal(colors, expected)


@pytest.mark.parametrize("name", ["Basic", pytest.param("Bonus", marks=pytest.mark.bonus)])
def test_hopfield_gui_loads_selected_pair_displays_and_browses(window, toy_data, name):
    tab = build_app(window, toy_data).hopfield
    assert name in tab.dataset["values"]
    tab.dataset.set(name)
    tab.run.invoke()
    targets = load_patterns(toy_data / f"hopfield/{name}_Training.txt")
    cues = load_patterns(toy_data / f"hopfield/{name}_Testing.txt")
    for index, (target, cue) in enumerate(zip(targets, cues, strict=True)):
        if index:
            tab.next_case.invoke()
        rendered = pixels(tab.canvas, "target", target.shape)
        palette = {v: rendered[target == v][0] for v in (-1, 1)}
        assert palette[-1] != palette[1]
        assert_image(rendered, target, palette)
        assert_image(pixels(tab.canvas, "cue", cue.shape), cue, palette)
        assert_image(pixels(tab.canvas, "recalled", target.shape), target, palette)


@pytest.mark.bonus
def test_noise_button_shows_new_cue_and_recalled_target(window, toy_data):
    folder = toy_data / "hopfield"
    (folder / "Basic_Testing.txt").write_bytes((folder / "Basic_Training.txt").read_bytes())
    tab = build_app(window, toy_data).hopfield
    tab.dataset.set("Basic")
    tab.run.invoke()
    original = pixels(tab.canvas, "target", (12, 9)).copy()
    np.testing.assert_array_equal(pixels(tab.canvas, "cue", original.shape), original)
    tab.noise.invoke()
    assert np.count_nonzero(pixels(tab.canvas, "cue", original.shape) != original) == 1
    np.testing.assert_array_equal(pixels(tab.canvas, "recalled", original.shape), original)


def test_som_gui_runs_selected_file_and_iteration_count(window, toy_data):
    tab = build_app(window, toy_data).som
    assert {"cloud.txt", "line.TXT"} <= set(tab.dataset["values"])
    for name, count, iterations in [("line.TXT", 2, 3), ("cloud.txt", 4, 7)]:
        tab.dataset.set(name)
        tab.iterations.delete(0, "end")
        tab.iterations.insert(0, str(iterations))
        tab.run.invoke()
        assert len(tab.canvas.find_withtag("sample")) == count
        assert len(tab.result.history) == iterations + 1
        assert len(tab.result.winners) == count


def test_som_gui_displays_actionable_iteration_error(window, toy_data):
    tab = build_app(window, toy_data).som
    tab.dataset.set("cloud.txt")
    tab.iterations.delete(0, "end")
    tab.iterations.insert(0, "0")
    tab.run.invoke()
    assert tab.status.cget("text") == "[SOM_ITERATIONS] 迭代次數必須為正整數。"


@pytest.mark.bonus
def test_som_gui_draws_grid_connections_and_advances_frame(window, toy_data):
    tab = build_app(window, toy_data).som
    tab.dataset.set("cloud.txt")
    tab.iterations.delete(0, "end")
    tab.iterations.insert(0, "4")
    tab.run.invoke()
    rows, cols = tab.result.history[0].shape[:2]
    def positions():
        nodes = tab.canvas.find_withtag("node")
        assert len(nodes) == rows * cols
        assert len(tab.canvas.find_withtag("edge")) == rows * (cols - 1) + cols * (rows - 1)
        return [tab.canvas.coords(node) for node in nodes]

    before = positions()
    tab.next_frame.invoke()
    assert positions() != before
