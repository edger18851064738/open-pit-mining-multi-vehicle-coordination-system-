import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
import matplotlib
import math
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 
# 使用实际数据（从日志中提取的）
# 1011号车辆数据
truck_1011_samples = [
    (0, 58.86, -15.81),
    (20, 43.82, -28.01),
    (40, 25.47, -23.62),
    (60, 18.92, -4.97),
    (80, 9.81, 12.32),
    (None, 1.90, 22.64)
]

# 1010号车辆数据
truck_1010_samples = [
    (0, 65.26, -22.74),
    (20, 49.73, -34.79),
    (40, 31.16, -29.80),
    (60, 20.61, -13.29),
    (80, 14.27, 5.62),
    (None, 2.50, 21.85)
]

# truck_1026号车辆数据
truck_1026_samples = [
    (0, 32.07, -62.33),
    (20, 21.60, -45.69),
    (40, 20.46, -26.71),
    (60, 18.98, -6.78),
    (80, 10.97, 10.70),
    (None, -1.10, 26.64),
    (120, -13.02, 42.70),
    (140, -24.94, 58.76),
    (160, -36.95, 74.76),
    (180, -48.97, 90.74),
    (200, -61.00, 106.72)
]

# 根据采样点生成完整路径数据
def generate_path_from_samples(samples, total_points):
    # 提取有索引的点
    indexed_samples = [(idx, x, y) for idx, x, y in samples if idx is not None]
    
    # 生成完整路径
    path = []
    for i in range(total_points):
        # 找到最近的两个采样点进行插值
        for j in range(len(indexed_samples)-1):
            idx1, x1, y1 = indexed_samples[j]
            idx2, x2, y2 = indexed_samples[j+1]
            
            if idx1 <= i <= idx2:
                # 线性插值
                ratio = (i - idx1) / (idx2 - idx1) if idx2 != idx1 else 0
                x = x1 + ratio * (x2 - x1)
                y = y1 + ratio * (y2 - y1)
                path.append((x, y))
                break
        else:
            # 处理最后一段或超出范围的点
            if i >= indexed_samples[-1][0]:
                # 使用最后一个采样点和终点进行插值
                idx_last, x_last, y_last = indexed_samples[-1]
                ratio = (i - idx_last) / (total_points - 1 - idx_last) if total_points - 1 != idx_last else 0
                x = x_last + ratio * (samples[-1][1] - x_last)
                y = y_last + ratio * (samples[-1][2] - y_last)
                path.append((x, y))
            else:
                # 使用起点和第一个采样点进行插值
                idx_first, x_first, y_first = indexed_samples[0]
                ratio = i / idx_first if idx_first != 0 else 0
                x = samples[0][1] + ratio * (x_first - samples[0][1])
                y = samples[0][2] + ratio * (y_first - samples[0][2])
                path.append((x, y))
    
    return path

truck_1011_points = generate_path_from_samples(truck_1011_samples, 94)
truck_1010_points = generate_path_from_samples(truck_1010_samples, 102)
truck_1026_points = generate_path_from_samples(truck_1026_samples, 201)

# 车辆尺寸参数
truck_length = 12.0  # 矿卡通常较大
truck_width = 6.0
safe_margin = 1.0

# 创建图形
fig, ax = plt.subplots(figsize=(14, 10))

# 绘制三辆车的路径
ax.plot([p[0] for p in truck_1011_points], [p[1] for p in truck_1011_points], 'r-', label='1011号车')
ax.plot([p[0] for p in truck_1010_points], [p[1] for p in truck_1010_points], 'g-', label='1010号车')
ax.plot([p[0] for p in truck_1026_points], [p[1] for p in truck_1026_points], 'b-', label='truck_1026')

# 标记起点和终点
def plot_start_end(points, color, label):
    ax.plot(points[0][0], points[0][1], f'{color}o', markersize=8)
    ax.text(points[0][0]+1, points[0][1]+1, f'{label}起点', fontsize=10)
    ax.plot(points[-1][0], points[-1][1], f'{color}*', markersize=10)
    ax.text(points[-1][0]+1, points[-1][1]+1, f'{label}终点', fontsize=10)

plot_start_end(truck_1011_points, 'r', '1011')
plot_start_end(truck_1010_points, 'g', '1010')
plot_start_end(truck_1026_points, 'b', '1026')

# 添加标题和图例
ax.set_title('矿卡路径分析 - 1011/1010/1026号车', fontsize=16)
ax.legend(loc='upper right')
ax.set_xlabel('X坐标 (m)')
ax.set_ylabel('Y坐标 (m)')
ax.grid(True)

# 设置坐标轴范围
margin = 20
ax.set_xlim(min(min([p[0] for p in truck_1011_points]), 
                min([p[0] for p in truck_1010_points]),
                min([p[0] for p in truck_1026_points])) - margin,
           max(max([p[0] for p in truck_1011_points]),
               max([p[0] for p in truck_1010_points]),
               max([p[0] for p in truck_1026_points])) + margin)
ax.set_ylim(min(min([p[1] for p in truck_1011_points]),
                min([p[1] for p in truck_1010_points]),
                min([p[1] for p in truck_1026_points])) - margin,
           max(max([p[1] for p in truck_1011_points]),
               max([p[1] for p in truck_1010_points]),
               max([p[1] for p in truck_1026_points])) + margin)

# 设置等比例坐标轴
ax.set_aspect('equal')

plt.tight_layout()
plt.show()