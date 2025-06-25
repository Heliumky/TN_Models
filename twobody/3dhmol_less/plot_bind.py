import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
base_dir = "."

results = []           # 用 list 就好了
R_values_sorted = []   # 可选：保存 R 值顺序供参考
R_subdirs = []         # 用来收集 (R_value, subdir_name)
E_values_sorted = [] 
# 1️⃣ 找到所有以 R 开头的资料夹，并取得 R 值
for subdir in os.listdir(base_dir):
    if subdir.startswith("R"):
        match = re.search(r"R([0-9\.]+)", subdir)
        if match:
            R_value = float(match.group(1))
            R_subdirs.append((R_value, subdir))

# 2️⃣ 按 R 值升序排列
R_subdirs.sort(key=lambda x: x[0])

# 3️⃣ 按顺序载入
for R_value, subdir in R_subdirs:
    pkl_file = os.path.join(base_dir, subdir, "3d_dmrg_N=6_par=0.pkl")
    if os.path.isfile(pkl_file):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        results.append(data)           # 按顺序放入
        R_values_sorted.append(R_value)  # 可选：保存 R 值
        E_values_sorted.append(data["Energy"]+1/R_value)
        print(f"已载入 R = {R_value} (index = {len(results) - 1})")

# 检查：
print("\nR 值顺序:", R_values_sorted)
print("\nE 值:", E_values_sorted)
print(f"共载入 {len(results)} 个文件，results[0] 对应 R = {R_values_sorted[0]}")

plt.plot(R_values_sorted, E_values_sorted,
         color='r',
         marker='+',
         markersize=6,
         linestyle=':',
         label='DMRG')


# 坐标轴标签
plt.xlabel('R (angstrom)')
plt.ylabel('Total Energy (Hartree)')

# 限制显示范围（按你的数据调整）
plt.xlim(0, 8)
plt.ylim(-1.25, -0.7)

# 打开图例
plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
plt.legend(frameon=False)

# 打开虚线网格（可选）
plt.grid(True, linestyle='--', alpha=0.3)

# 紧凑布局
plt.tight_layout()

# 显示图像
plt.show()
