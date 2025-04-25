import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
import matplotlib
import math
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 
# 使用实际数据（从日志中提取的）
# 1007号车辆的路径数据
truck_1007_start = (-13.18, 22.20)
truck_1007_end = (41.19, -48.60)
# 采样点数据
truck_1007_samples = [
    (0, -13.18, 22.20),
    (20, -1.43, 6.02),
    (40, 12.77, -5.89),
    (60, 32.20, -8.22),
    (80, 45.92, -22.48),
    (None, 46.03, -41.40)  # 最后一个点没有索引
]

# 1006号车辆的路径数据
truck_1006_start = (-13.18, 22.19)
truck_1006_end = (60.21, -13.63)
# 采样点数据
truck_1006_samples = [
    (0, -13.18, 22.19),
    (20, -1.43, 6.02),
    (40, 12.78, -5.89),
    (60, 32.21, -8.17),
    (80, 42.16, -24.25),
    (None, 34.52, -42.22),
    (120, 46.11, -29.20),
    (140, 59.50, -14.34)
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
truck_1007_points = generate_path_from_samples(truck_1007_samples, 110)
truck_1006_points = generate_path_from_samples(truck_1006_samples, 142)

# 车辆尺寸参数
truck_length = 12.0  # 矿卡通常较大
truck_width = 6.0
safe_margin = 1.0

# 创建图形
fig, ax = plt.subplots(figsize=(14, 10))

# 绘制两辆车的路径
ax.plot([p[0] for p in truck_1007_points], [p[1] for p in truck_1007_points], 'g-', label='1007号车(passing车辆)')
ax.plot([p[0] for p in truck_1006_points], [p[1] for p in truck_1006_points], 'b-', label='1006号车(降级车辆)')

# 标记路径起点和终点
ax.plot(truck_1007_start[0], truck_1007_start[1], 'go', markersize=8)
ax.text(truck_1007_start[0]+1, truck_1007_start[1]+1, '1007起点', fontsize=10)
ax.plot(truck_1007_end[0], truck_1007_end[1], 'g*', markersize=10)
ax.text(truck_1007_end[0]+1, truck_1007_end[1]+1, '1007终点', fontsize=10)

ax.plot(truck_1006_start[0], truck_1006_start[1], 'bo', markersize=8)
ax.text(truck_1006_start[0]-10, truck_1006_start[1]+1, '1006起点', fontsize=10)
ax.plot(truck_1006_end[0], truck_1006_end[1], 'b*', markersize=10)
ax.text(truck_1006_end[0]+1, truck_1006_end[1]+1, '1006终点', fontsize=10)

# 绘制采样点
for _, x, y in truck_1007_samples:
    ax.plot(x, y, 'gx', markersize=8)
for _, x, y in truck_1006_samples:
    ax.plot(x, y, 'bx', markersize=8)

ax.plot([], [], 'gx', markersize=8, label='1007采样点')
ax.plot([], [], 'bx', markersize=8, label='1006采样点')

# 找出最可能的冲突点
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

min_dist = float('inf')
conflict_point_1007 = None
conflict_point_1006 = None
conflict_idx_1007 = -1
conflict_idx_1006 = -1

for i, p1 in enumerate(truck_1007_points):
    for j, p2 in enumerate(truck_1006_points):
        dist = distance(p1, p2)
        if dist < min_dist:
            min_dist = dist
            conflict_point_1007 = p1
            conflict_point_1006 = p2
            conflict_idx_1007 = i
            conflict_idx_1006 = j

# 检查冲突点是否被步长跳过
is_detected = False
sample_indices_1007 = [idx for idx, _, _ in truck_1007_samples if idx is not None]
for idx in sample_indices_1007:
    if abs(idx - conflict_idx_1007) < 5:  # 允许一定容差
        is_detected = True
        break

# 绘制最可能的冲突点及矩形框
if conflict_point_1007 and conflict_point_1006:
    ax.plot(conflict_point_1007[0], conflict_point_1007[1], 'mx', markersize=12)
    ax.plot(conflict_point_1006[0], conflict_point_1006[1], 'mx', markersize=12)
    ax.plot([conflict_point_1007[0], conflict_point_1006[0]], 
            [conflict_point_1007[1], conflict_point_1006[1]], 'm--', linewidth=2)
    
    # 计算车辆航向角
    def calculate_heading(points, idx):
        if idx < len(points) - 1:
            dx = points[idx+1][0] - points[idx][0]
            dy = points[idx+1][1] - points[idx][1]
        else:
            dx = points[idx][0] - points[idx-1][0]
            dy = points[idx][1] - points[idx-1][1]
        return math.degrees(math.atan2(dy, dx))
    
    angle_1007 = calculate_heading(truck_1007_points, conflict_idx_1007)
    angle_1006 = calculate_heading(truck_1006_points, conflict_idx_1006)
    
    # 绘制车辆矩形框
    def draw_truck_rectangle(point, angle, color):
        cos_angle = math.cos(math.radians(angle))
        sin_angle = math.sin(math.radians(angle))
        
        # 生成矩形四个角点
        corners = []
        for dx, dy in [(-truck_length/2, -truck_width/2), 
                       (truck_length/2, -truck_width/2),
                       (truck_length/2, truck_width/2),
                       (-truck_length/2, truck_width/2)]:
            # 旋转角点并加上安全边界
            x_rot = dx * cos_angle - dy * sin_angle
            y_rot = dx * sin_angle + dy * cos_angle
            corners.append((point[0] + x_rot, point[1] + y_rot))
        
        # 绘制矩形
        polygon = Polygon(corners, fill=False, edgecolor=color, alpha=0.7, linewidth=2)
        ax.add_patch(polygon)
        
        # 绘制带安全边界的矩形
        safety_corners = []
        safe_length = truck_length + 2 * safe_margin
        safe_width = truck_width + 2 * safe_margin
        for dx, dy in [(-safe_length/2, -safe_width/2), 
                      (safe_length/2, -safe_width/2),
                      (safe_length/2, safe_width/2),
                      (-safe_length/2, safe_width/2)]:
            # 旋转角点
            x_rot = dx * cos_angle - dy * sin_angle
            y_rot = dx * sin_angle + dy * cos_angle
            safety_corners.append((point[0] + x_rot, point[1] + y_rot))
        
        # 绘制安全边界
        polygon = Polygon(safety_corners, fill=False, edgecolor=color, alpha=0.4, linewidth=1, linestyle='--')
        ax.add_patch(polygon)
    
    # 绘制车辆矩形
    draw_truck_rectangle(conflict_point_1007, angle_1007, 'g')
    draw_truck_rectangle(conflict_point_1006, angle_1006, 'b')
    
    # 标记最小距离和冲突点信息
    ax.text((conflict_point_1007[0] + conflict_point_1006[0])/2, 
            (conflict_point_1007[1] + conflict_point_1006[1])/2 + 5, 
            f'最小距离: {min_dist:.2f}m', fontsize=12, ha='center')
    
    # 标记是否被步长跳过
    if not is_detected:
        ax.text((conflict_point_1007[0] + conflict_point_1006[0])/2, 
                (conflict_point_1007[1] + conflict_point_1006[1])/2 - 5, 
                '⚠️ 此冲突点被步长跳过!', fontsize=14, ha='center', 
                bbox=dict(facecolor='yellow', alpha=0.5))

# 添加标题和图例
ax.set_title('路径变化与碰撞检测分析 - 1007号车与1006号车', fontsize=16)
ax.legend(loc='upper right')
ax.set_xlabel('X坐标 (m)')
ax.set_ylabel('Y坐标 (m)')
ax.grid(True)

# 添加说明文本
explanation = (
    "分析说明:\n"
    "1. 绿线: 1007号降级车辆路径(110点)\n"
    "2. 蓝线: 1006号passing车辆路径(142点)\n"
    "3. X标记: 路径采样点\n"
    "4. 品红色X: 两车最接近的潜在碰撞点\n"
    "5. 虚线矩形: 车辆安全边界\n" +
    (f"6. ⚠️注意: 最小距离点({min_dist:.2f}m)被步长跳过!" if not is_detected else "")
)
ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=12, 
        verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.7))

# 设置坐标轴范围确保可以看到完整路径
margin = 10
ax.set_xlim(min(min([p[0] for p in truck_1007_points]), min([p[0] for p in truck_1006_points])) - margin,
           max(max([p[0] for p in truck_1007_points]), max([p[0] for p in truck_1006_points])) + margin)
ax.set_ylim(min(min([p[1] for p in truck_1007_points]), min([p[1] for p in truck_1006_points])) - margin,
           max(max([p[1] for p in truck_1007_points]), max([p[1] for p in truck_1006_points])) + margin)

# 设置等比例坐标轴
ax.set_aspect('equal')

plt.tight_layout()
plt.show()