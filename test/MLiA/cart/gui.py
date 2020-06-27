from MLiA.cart import regTrees
import numpy as np
import tkinter as tk
import matplotlib
# 设定后端为 TkAgg
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tolS, tolN):
    # 清空之前的图像
    reDraw.f.clf()
    # 重新添加子图
    reDraw.a = reDraw.f.add_subplot(111)
    # 检测选框是否选中，确定是模型树还是回归树
    if chk_btn_var.get():
        if tolN < 2:
            tolN = 2
        my_tree = regTrees.createTree(reDraw.raw_data, regTrees.modelLeaf, regTrees.modelErr, (tolS, tolN))
        y_hat = regTrees.createForecast(my_tree, reDraw.test_data, regTrees.modelTreeEval)
    else:
        # 回归树
        my_tree = regTrees.createTree(reDraw.raw_data, ops=(tolS, tolN))
        y_hat = regTrees.createForecast(my_tree, reDraw.test_data)
    # 画真实点的散点图
    reDraw.a.scatter(np.array(reDraw.raw_data[:, 0]), np.array(reDraw.raw_data[:, 1]), s=5)
    # 画预测值的直线图
    reDraw.a.plot(reDraw.test_data, y_hat, linewidth=2.0)
    reDraw.canvas.draw()


def getInputs():
    # 获取用户输入值，tolN为整型，tolS为浮点型
    try:
        tol_n = int(tol_n_entry.get())
    except:
        tol_n = 10
        print("enter Integer for tolN")
        tol_n_entry.delete(0, tk.END)
        tol_n_entry.insert(0, '10')

    try:
        tol_s = float(tol_s_entry.get())
    except:
        tol_s = 1.0
        print("enter Float for tolS")
        tol_s_entry.delete(0, tk.END)
        tol_s_entry.insert(0, '1.0')
    return tol_n, tol_s


def drawNewTree():
    tol_n, tol_s = getInputs()
    reDraw(tol_s, tol_n)


root = tk.Tk()

reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

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

reDraw.raw_data = np.mat(regTrees.loadDataSet('data/sine.txt'))
reDraw.test_data = np.arange(np.min(reDraw.raw_data[:, 0]), np.max(reDraw.raw_data[:, 0]), 0.01)

reDraw(1.0, 10)

if __name__ == '__main__':
    root.mainloop()
