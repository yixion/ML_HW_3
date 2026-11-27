"""開發入口。封裝時一併納入 src/hw3 與 data。"""
from pathlib import Path
import sys
import tkinter as tk

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from mem_storage_machine.gui import build_app


def main():
    root = tk.Tk()
    root.title("Hopfield 與 SOM")
    app = build_app(root, BASE / "data")
    root.mainloop()


if __name__ == "__main__":
    main()
