import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# ------------------ 参数设定 ------------------

# 真目标参数
TARGET_CENTER = np.array([0, 200, 0])  # 真目标下底面圆心
TARGET_RADIUS = 7
TARGET_HEIGHT = 10

# 导弹参数
MISSILE_SPEED = 300  # m/s

# 无人机参数
UAV_SPEED_MIN = 70   # m/s
UAV_SPEED_MAX = 140  # m/s
UAV_ALTITUDE = None  # 将在初始化时根据初始位置设定

# 烟幕弹参数
SHELL_DESCENT_SPEED = 3  # m/s 下沉速度
EFFECTIVE_RADIUS = 10    # m 有效遮蔽半径
EFFECTIVE_DURATION = 20  # s 有效遮蔽持续时间
MIN_INTER_SHELL_TIME = 1 # s 同一无人机两次投放最小间隔

# 初始状态 (t=0时刻)
# 导弹位置 (x, y, z)
missiles_initial = {
    'M1': np.array([20000, 0, 2000]),
    'M2': np.array([19000, 600, 2100]),
    'M3': np.array([18000, -600, 1900]),
}
# 无人机位置 (x, y, z)
uavs_initial = {
    'FY1': np.array([17800, 0, 1800]),
    'FY2': np.array([12000, 1400, 1400]),
    'FY3': np.array([6000, -3000, 700]),
    'FY4': np.array([11000, 2000, 1800]),
    'FY5': np.array([13000, -2000, 1300]),
}

# 假设所有无人机在同一高度，取FY1的高度作为标准
UAV_ALTITUDE = uavs_initial['FY1'][2]

# 计算导弹到假目标(0,0,0)的距离和飞行时间
missile_times_to_target = {}
for name, pos in missiles_initial.items():
    distance = np.linalg.norm(pos - np.array([0, 0, 0]))
    time = distance / MISSILE_SPEED
    missile_times_to_target[name] = time

# ------------------ 模型与仿真 ------------------

class Shell:
    """烟幕弹类"""
    def __init__(self, deploy_time, deploy_pos, uav_velocity):
        self.deploy_time = deploy_time
        self.deploy_pos = np.array(deploy_pos)
        # 烟幕弹初始速度等于无人机速度
        self.initial_velocity = np.array(uav_velocity)
        self.burst_time = None
        self.burst_pos = None

    def set_burst(self, burst_time):
        """设定起爆时间和位置"""
        self.burst_time = burst_time
        # 起爆时烟幕弹位置 = 投放位置 + 无人机速度 * (起爆时间 - 投放时间)
        self.burst_pos = self.deploy_pos + self.initial_velocity * (self.burst_time - self.deploy_time)

    def get_cloud_position(self, time):
        """获取烟幕云团在给定时间的位置 (云团中心)"""
        if self.burst_time is None or time < self.burst_time:
            return None # 尚未起爆
        # 云团起爆后匀速下沉
        descent_distance = SHELL_DESCENT_SPEED * (time - self.burst_time)
        cloud_center = self.burst_pos - np.array([0, 0, descent_distance])
        return cloud_center

    def is_effective_at_time(self, time, target_point):
        """判断在给定时间点，烟幕云团是否对目标点有效遮蔽"""
        if self.burst_time is None or time < self.burst_time or time > (self.burst_time + EFFECTIVE_DURATION):
            return False # 时间不在有效期内
        cloud_center = self.get_cloud_position(time)
        if cloud_center is None:
            return False
        distance = np.linalg.norm(cloud_center - target_point)
        return distance <= EFFECTIVE_RADIUS

class UAV:
    """无人机类"""
    def __init__(self, name, initial_pos):
        self.name = name
        self.initial_pos = np.array(initial_pos)
        self.flight_direction = None # 单位向量
        self.flight_speed = None # m/s
        self.shells = []

    def set_flight_plan(self, direction, speed):
        """设定飞行计划"""
        # 确保是水平方向
        dir_2d = np.array([direction[0], direction[1], 0])
        norm = np.linalg.norm(dir_2d[:2])
        if norm > 1e-10:
            self.flight_direction = dir_2d / norm
        else:
            # 默认方向以防除零
            self.flight_direction = np.array([1, 0, 0])
        self.flight_speed = max(UAV_SPEED_MIN, min(UAV_SPEED_MAX, speed))

    def get_position(self, time):
        """获取无人机在给定时间的位置"""
        return self.initial_pos + self.flight_direction * self.flight_speed * time

    def deploy_shell(self, deploy_time):
        """在指定时间投放一枚烟幕弹"""
        deploy_pos = self.get_position(deploy_time)
        shell = Shell(deploy_time, deploy_pos, self.flight_direction * self.flight_speed)
        self.shells.append(shell)
        return shell

def calculate_missile_position(missile_name, time):
    """计算导弹在给定时间的位置"""
    initial_pos = missiles_initial[missile_name]
    direction = np.array([0, 0, 0]) - initial_pos # 指向假目标(0,0,0)
    unit_direction = direction / np.linalg.norm(direction)
    distance_traveled = MISSILE_SPEED * time
    current_pos = initial_pos + unit_direction * distance_traveled
    return current_pos

def total_effective_time(uavs_dict, target_point=TARGET_CENTER, t_start=0, t_end=100, dt=0.2):
    """计算所有烟幕弹对真目标的总有效遮蔽时间"""
    effective_duration = 0.0
    t = t_start
    while t <= t_end:
        is_covered = False
        for uav in uavs_dict.values():
            for shell in uav.shells:
                if shell.is_effective_at_time(t, target_point):
                    is_covered = True
                    break
            if is_covered:
                break
        if is_covered:
            effective_duration += dt
        t += dt
    return effective_duration


# ------------------ 优化策略 ------------------

def objective_function_for_uav(variables, uav_name, uavs_dict_template, missile_name):
    """
    优化目标函数：最小化导弹看到真目标的时间（即最大化遮蔽时间）
    对于单个无人机投放两枚弹的简单策略优化。
    variables: [direction_x, direction_y, speed, t_deploy1, t_burst1, t_deploy2, t_burst2]
    """
    # 深拷贝模板以避免修改原始字典
    import copy
    uavs_dict = copy.deepcopy(uavs_dict_template)
    
    # 解析变量
    dir_x, dir_y, speed, t_dep1, t_burst1, t_dep2, t_burst2 = variables

    # 设置无人机
    uav = uavs_dict[uav_name]
    uav.set_flight_plan([dir_x, dir_y], speed)

    # 投放和起爆
    # 确保时间顺序和间隔
    if t_dep2 <= t_dep1 + MIN_INTER_SHELL_TIME:
        t_dep2 = t_dep1 + MIN_INTER_SHELL_TIME + 1e-6

    shell1 = uav.deploy_shell(max(0, t_dep1))
    shell1.set_burst(max(t_dep1, t_burst1))

    shell2 = uav.deploy_shell(max(t_dep1 + MIN_INTER_SHELL_TIME + 1e-6, t_dep2))
    shell2.set_burst(max(t_dep2, t_burst2))

    # 计算导弹飞行时间
    missile_time = missile_times_to_target[missile_name]
    
    # 优化目标：最大化导弹来袭时间内被遮蔽的时间
    # 仿真时间窗口
    t_start = max(0, missile_time - 30)
    t_end = missile_time + 10
    
    covered_time = total_effective_time(uavs_dict, TARGET_CENTER, t_start, t_end, dt=0.5)
    # 目标是最小化未被遮蔽的时间
    uncovered_time = (t_end - t_start) - covered_time
    # 添加一个惩罚项，如果起爆时间早于投放时间
    penalty = 0
    if t_burst1 < t_dep1: penalty += (t_dep1 - t_burst1) * 1000
    if t_burst2 < t_dep2: penalty += (t_dep2 - t_burst2) * 1000
    return uncovered_time + penalty

def optimize_single_uav_single_missile(uav_name, missile_name, uavs_dict_template):
    """为单个无人机针对单个导弹优化投放策略"""
    print(f"正在为无人机 {uav_name} 针对导弹 {missile_name} 优化策略...")
    
    # 初始猜测值
    # 基于初始距离和时间做一个粗略估计
    uav_pos = uavs_initial[uav_name]
    missile_pos = missiles_initial[missile_name]
    initial_direction_guess = missile_pos[:2] - uav_pos[:2] # 指向导弹初始水平位置
    initial_speed_guess = 100
    
    initial_guess = [
        initial_direction_guess[0], initial_direction_guess[1], # 方向 (x, y)
        initial_speed_guess,  # 速度
        10, 25, # 第一枚弹 投放和起爆时间
        30, 45  # 第二枚弹 投放和起爆时间
    ]

    # 约束条件
    missile_time = missile_times_to_target[missile_name]
    bounds = [
        (-10000, 10000), (-10000, 10000), # 方向 (无界，优化器会处理)
        (UAV_SPEED_MIN, UAV_SPEED_MAX), # 速度
        (0, missile_time + 20), (0, missile_time + 40), # 第一枚时间
        (0, missile_time + 40), (0, missile_time + 60) # 第二枚时间
    ]
    
    result = minimize(objective_function_for_uav, initial_guess, args=(uav_name, uavs_dict_template, missile_name),
                      method='L-BFGS-B', bounds=bounds, options={'disp': False})

    if result.success:
        opt_vars = result.x
        dir_x, dir_y, speed, t_dep1, t_burst1, t_dep2, t_burst2 = opt_vars
        speed_clipped = max(UAV_SPEED_MIN, min(UAV_SPEED_MAX, speed))
        print(f"优化成功:")
        print(f"  方向({dir_x:.2f}, {dir_y:.2f}), 速度 {speed_clipped:.2f} m/s")
        print(f"  弹1: 投放 @{t_dep1:.2f}s, 起爆 @{t_burst1:.2f}s")
        print(f"  弹2: 投放 @{t_dep2:.2f}s, 起爆 @{t_burst2:.2f}s")
        print(f"  预估未遮蔽时间: {result.fun:.2f}s")
        
        # 返回优化后的UAV实例
        import copy
        uav_opt = copy.deepcopy(uavs_dict_template[uav_name])
        uav_opt.set_flight_plan([dir_x, dir_y], speed_clipped)
        if t_dep2 <= t_dep1 + MIN_INTER_SHELL_TIME:
            t_dep2 = t_dep1 + MIN_INTER_SHELL_TIME + 1e-6
        shell1 = uav_opt.deploy_shell(max(0, t_dep1))
        shell1.set_burst(max(t_dep1, t_burst1))
        shell2 = uav_opt.deploy_shell(max(t_dep1 + MIN_INTER_SHELL_TIME + 1e-6, t_dep2))
        shell2.set_burst(max(t_dep2, t_burst2))
        
        return uav_opt
    else:
        print(f"优化失败: {result.message}")
        return None

# ------------------ 可视化 ------------------

def draw_sphere(ax, center, radius, color, alpha):
    """在3D图上绘制一个球体"""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def visualize_scenario(uavs_dict, missile_names_to_plot=['M1', 'M2', 'M3'], show_clouds_at_time=None):
    """可视化场景和部分轨迹"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制真目标 (圆柱)
    z_target = np.linspace(TARGET_CENTER[2], TARGET_CENTER[2] + TARGET_HEIGHT, 50)
    theta_target = np.linspace(0, 2 * np.pi, 50)
    Theta_target, Z_target = np.meshgrid(theta_target, z_target)
    X_target = TARGET_CENTER[0] + TARGET_RADIUS * np.cos(Theta_target)
    Y_target = TARGET_CENTER[1] + TARGET_RADIUS * np.sin(Theta_target)
    ax.plot_surface(X_target, Y_target, Z_target, color='blue', alpha=0.3)
    # ax.plot_surface 有时会使图例复杂化，我们手动添加图例项
    target_patch = mpatches.Patch(color='blue', alpha=0.3, label='真目标')

    # 绘制假目标 (点)
    ax.scatter([0], [0], [0], color='red', s=50)
    fake_target = ax.scatter([0], [0], [0], color='red', s=0) # 用于图例
    fake_target.set_label('假目标')

    # 绘制初始导弹位置
    missile_colors = {'M1': 'green', 'M2': 'orange', 'M3': 'purple'}
    for name, pos in missiles_initial.items():
        ax.scatter(*pos, marker='^', s=40, color=missile_colors.get(name, 'black'))
    
    # 绘制初始无人机位置
    uav_colors = {'FY1': 'cyan', 'FY2': 'magenta', 'FY3': 'yellow', 'FY4': 'lime', 'FY5': 'pink'}
    for name, pos in uavs_initial.items():
        ax.scatter(*pos, marker='o', s=40, color=uav_colors.get(name, 'gray'))

    # 绘制导弹轨迹 (简化)
    t_max = max(missile_times_to_target.values()) + 10
    for name in missile_names_to_plot:
        t_vals = np.linspace(0, missile_times_to_target[name], 100)
        traj_x, traj_y, traj_z = [], [], []
        for t in t_vals:
            p = calculate_missile_position(name, t)
            traj_x.append(p[0])
            traj_y.append(p[1])
            traj_z.append(p[2])
        ax.plot(traj_x, traj_y, traj_z, linestyle='--', linewidth=1, color=missile_colors[name], label=f'{name}轨迹')

    # 绘制无人机轨迹和烟幕弹投放/起爆点
    t_traj_max = t_max
    uav_handles = []
    for uav in uavs_dict.values():
        if not uav.shells: continue
        # 无人机轨迹
        t_uav_vals = np.linspace(0, t_traj_max, 200)
        uav_traj_x, uav_traj_y, uav_traj_z = [], [], []
        for t in t_uav_vals:
            p = uav.get_position(t)
            uav_traj_x.append(p[0])
            uav_traj_y.append(p[1])
            uav_traj_z.append(p[2])
        line_uav = ax.plot(uav_traj_x, uav_traj_y, uav_traj_z, linewidth=1.5, color=uav_colors.get(uav.name, 'gray'))
        # 为图例创建一个不可见的点
        uav_handle = ax.scatter([0], [0], [0], color=uav_colors.get(uav.name, 'gray'), s=0)
        uav_handle.set_label(f'{uav.name}轨迹')
        uav_handles.append(uav_handle)

        # 投放点
        dep_points = np.array([s.deploy_pos for s in uav.shells])
        if len(dep_points) > 0:
            ax.scatter(dep_points[:, 0], dep_points[:, 1], dep_points[:, 2], marker='x', s=50, color='orange')
            dep_handle = ax.scatter([0], [0], [0], marker='x', s=50, color='orange')
            dep_handle.set_label(f'{uav.name}投放点')

        # 起爆点
        burst_points = np.array([s.burst_pos for s in uav.shells if s.burst_pos is not None])
        if len(burst_points) > 0:
            ax.scatter(burst_points[:, 0], burst_points[:, 1], burst_points[:, 2], marker='*', s=60, color='purple')
            burst_handle = ax.scatter([0], [0], [0], marker='*', s=60, color='purple')
            burst_handle.set_label(f'{uav.name}起爆点')

        # 烟幕云团轨迹示意图 (仅在起爆后一段时间)
        for shell in uav.shells:
         if shell.burst_time is not None:
            t_cloud = np.linspace(shell.burst_time, min(shell.burst_time + EFFECTIVE_DURATION, t_traj_max), 50)
            cloud_x, cloud_y, cloud_z = [], [], []
            for t in t_cloud:
                p = shell.get_cloud_position(t)
                if p is not None:
                    cloud_x.append(p[0])
                    cloud_y.append(p[1])
                    cloud_z.append(p[2])
            ax.plot(cloud_x, cloud_y, cloud_z, linewidth=1, alpha=0.4, color='gray') # 烟幕云轨迹

    # 在特定时间绘制烟幕云团有效范围 (球体)
    if show_clouds_at_time is not None:
        print(f"可视化时间点: {show_clouds_at_time:.2f}s")
        for uav in uavs_dict.values():
            for shell in uav.shells:
                if shell.burst_time is not None and \
                   shell.burst_time <= show_clouds_at_time <= (shell.burst_time + EFFECTIVE_DURATION):
                    cloud_center = shell.get_cloud_position(show_clouds_at_time)
                    if cloud_center is not None:
                        is_effective = shell.is_effective_at_time(show_clouds_at_time, TARGET_CENTER)
                        color = 'green' if is_effective else 'red'
                        draw_sphere(ax, cloud_center, EFFECTIVE_RADIUS, color, 0.2)
                        # 添加文字标签
                        ax.text(cloud_center[0], cloud_center[1], cloud_center[2], 
                                f'{uav.name[-1]}S{uav.shells.index(shell)+1}', 
                                color=color, fontsize=8)


    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('烟幕干扰弹投放策略仿真')

    # 手动构建图例
    handles = [target_patch, fake_target] + \
              [mpatches.Patch(color=missile_colors[n], label=f'{n}轨迹') for n in missile_names_to_plot] + \
              [h for h in uav_handles] + \
              [mpatches.Patch(color='orange', label='投放点'), 
               mpatches.Patch(color='purple', label='起爆点*')]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------ 主程序 ------------------

if __name__ == "__main__":
    print("--- 烟幕干扰弹投放策略仿真 ---")
    print("导弹飞行时间 (s):")
    for name, time in missile_times_to_target.items():
        print(f"  {name}: {time:.2f}")

    # 创建初始无人机字典模板
    uavs_template = {name: UAV(name, pos) for name, pos in uavs_initial.items()}

    # --- 示例 1: 为FY1无人机针对M1导弹进行优化 ---
    print("\n--- 示例 1: 优化单个无人机 FY1 对 M1 ---")
    optimized_uav_FY1 = optimize_single_uav_single_missile('FY1', 'M1', uavs_template)
    
    if optimized_uav_FY1:
        # 创建一个只包含优化后FY1的字典用于可视化
        uavs_for_viz = {name: uav for name, uav in uavs_template.items() if name != 'FY1'}
        uavs_for_viz['FY1'] = optimized_uav_FY1
        
        # 计算总有效遮蔽时间 (在整个相关时间段内)
        t_eval_start = max(0, missile_times_to_target['M1'] - 30)
        t_eval_end = missile_times_to_target['M1'] + 10
        total_time = total_effective_time(uavs_for_viz, TARGET_CENTER, t_eval_start, t_eval_end, dt=0.1)
        print(f"优化后，FY1对M1干扰下，在评估时间段内的有效遮蔽时间: {total_time:.2f} s")

        # 可视化 (在M1到达时)
        visualize_scenario(uavs_for_viz, missile_names_to_plot=['M1'], show_clouds_at_time=missile_times_to_target['M1'])


    # --- 示例 2: 简单的多无人机任务分配与优化 ---
    print("\n--- 示例 2: 简化多无人机协同优化 ---")
    # 简单策略：每枚导弹分配一架最近的无人机
    # M1 -> FY1, M2 -> FY2, M3 -> FY3 (基于初始x坐标接近度)
    uav_missile_pairs = [('FY1', 'M1'), ('FY2', 'M2'), ('FY3', 'M3')]
    
    # 存储优化后的无人机
    optimized_uavs_multi = {}
    for uav_name, missile_name in uav_missile_pairs:
        opt_uav = optimize_single_uav_single_missile(uav_name, missile_name, uavs_template)
        if opt_uav:
            optimized_uavs_multi[uav_name] = opt_uav
        else:
            # 如果优化失败，使用模板中的原始无人机
            optimized_uavs_multi[uav_name] = uavs_template[uav_name]
    
    # 对于未分配任务的无人机(FY4, FY5)，为了可视化，也加入字典
    for name in uavs_initial.keys():
        if name not in optimized_uavs_multi:
             optimized_uavs_multi[name] = uavs_template[name]

    # 计算在整个仿真时间内的总有效遮蔽时间
    t_global_start = 0
    t_global_end = max(missile_times_to_target.values()) + 20
    total_time_multi = total_effective_time(optimized_uavs_multi, TARGET_CENTER, t_global_start, t_global_end, dt=0.5)
    print(f"多无人机协同优化后，对真目标的总有效遮蔽时间: {total_time_multi:.2f} s")

    # 可视化多无人机场景 (在M3到达时，因为它最后到达)
    visualize_scenario(optimized_uavs_multi, missile_names_to_plot=['M1', 'M2', 'M3'], show_clouds_at_time=missile_times_to_target['M3'])








