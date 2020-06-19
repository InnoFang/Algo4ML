from MLiA.cart import regTrees, dataset
import numpy as np
import tkinter as tk
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tolS, tolN):
    pass


def drawNewTree():
    pass


root = tk.Tk()

tk.Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
tk.Label(root, text="tolN").grid(row=1, column=0)
tk.Label(root, text="tolS").grid(row=2, column=0)

tol_n_entry = tk.Entry(root)
tol_n_entry.grid(row=1, column=1)
tol_n_entry.insert(0, '10')

tol_s_entry = tk.Entry(root)
tol_s_entry.grid(row=2, column=1)
tol_s_entry.insert(0, '1.0')

tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)

chk_btn_var = tk.IntVar()
chk_btn = tk.Checkbutton(root, text="Model Tree", variable=chk_btn_var)
chk_btn.grid(row=3, column=0, columnspan=2)

reDraw.raw_data = np.mat(dataset.load_sine())
reDraw.test_data = np.arange(np.min(reDraw.raw_data[:, 0]), np.max(reDraw.raw_data[:, 0]), 0.01)

reDraw(1.0, 10)

root.mainloop()
