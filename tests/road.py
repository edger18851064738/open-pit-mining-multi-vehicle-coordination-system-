import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False 

# 从日志中提取的车辆路径数据
# 1007号车辆数据
truck_1007_samples = [
    (0, 58.78, -15.39),
    (20, 43.84, -28.02),
    (40, 25.47, -23.62),
    (60, 18.92, -4.97),
    (80, 9.81, 12.32),
    (100, -2.30, 28.25),
    (None, -11.24, 40.29)  # 终点
]

# 1006号车辆数据
truck_1006_samples = [
    (0, 66.07, -21.27),
    (20, 50.69, -34.54),
    (40, 31.96, -30.41),
    (60, 20.85, -14.26),
    (80, 14.80, 4.73),
    (100, 3.11, 21.05),
    (120, -8.86, 37.08),
    (None, -11.24, 40.29)  # 终点
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

# 生成完整路径
truck_1007_points = generate_path_from_samples(truck_1007_samples, 116)
truck_1006_points = generate_path_from_samples(truck_1006_samples, 125)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制路径
plt.plot([p[0] for p in truck_1007_points], [p[1] for p in truck_1007_points], 'r-', label='1007号车')
plt.plot([p[0] for p in truck_1006_points], [p[1] for p in truck_1006_points], 'g-', label='1006号车')

# 绘制起点和终点
def plot_start_end(points, color, label):
    plt.plot(points[0][0], points[0][1], f'{color}o', markersize=8)
    plt.text(points[0][0]+1, points[0][1]+1, f'{label}起点', fontsize=10)
    plt.plot(points[-1][0], points[-1][1], f'{color}*', markersize=10)
    plt.text(points[-1][0]+1, points[-1][1]+1, f'{label}终点', fontsize=10)

plot_start_end(truck_1007_points, 'r', '1007')
plot_start_end(truck_1006_points, 'g', '1006')

# 标记碰撞点
collision_x1, collision_y1 = 5.55, 17.89
collision_x2, collision_y2 = -0.50, 25.84
plt.plot([collision_x1, collision_x2], [collision_y1, collision_y2], 'ko-', label='碰撞区域')
plt.text(collision_x1, collision_y1, '碰撞点1', fontsize=10)
plt.text(collision_x2, collision_y2, '碰撞点2', fontsize=10)

# 设置图形属性
plt.grid(True)
plt.axis('equal')
plt.xlabel('X坐标 (m)')
plt.ylabel('Y坐标 (m)')
plt.title('矿卡路径分析与碰撞检测', fontsize=14)
plt.legend(loc='upper right')

# 添加uuid信息
plt.text(0.02, 0.02, 'uuid: 5759ceb9bbd745dc9752db5347660aa9', fontsize=8, transform=plt.gcf().transFigure)

plt.tight_layout()
plt.show()