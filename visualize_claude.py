import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


speed_missle=300
target_true_lower_surface=(0,200,0)
M1_init_pos=(20000,0,2000)
M2_init_pos=(19000,600,2100)
M3_init_pos=(18000,-600,1900)
FY1_init_pos=(17800,0,1800)
FY2_init_pos=(12000,1400,1400)
FY3_init_pos=(6000,-3000,700)
FY4_init_pos=(11000,2000,1800)
FY5_init_pos=(13000,-2000,1300)


# --- 1. 补充缺失的 Trajectory 类 ---
class Trajectory:
    """
    存储和管理轨迹数据的类。
    """
    def __init__(self):
        self.times = []
        self.positions = []

    def add_point(self, time, position):
        """添加一个轨迹点"""
        self.times.append(time)
        self.positions.append(np.array(position, dtype=float))

    def get_positions_array(self):
        """获取位置点的NumPy数组"""
        return np.array(self.positions)

    def get_position_at_time(self, t):
        """
        根据时间线性插值获取位置。
        假设轨迹点是按时间顺序添加的。
        """
        if not self.times:
            raise ValueError("Trajectory is empty.")
        if len(self.times) == 1:
             return self.positions[0]

        # 如果时间在轨迹范围外，返回端点
        if t <= self.times[0]:
            return self.positions[0]
        if t >= self.times[-1]:
            return self.positions[-1]

        # 找到时间区间
        for i in range(len(self.times) - 1):
            t1, t2 = self.times[i], self.times[i+1]
            if t1 <= t <= t2:
                p1, p2 = self.positions[i], self.positions[i+1]
                # 线性插值
                ratio = (t - t1) / (t2 - t1) if (t2 - t1) > 0 else 0
                return p1 + ratio * (p2 - p1)
        # 理论上不会到达这里，因为上面已经处理了边界
        return self.positions[-1] # Fallback

# --- 2. 补充/修正的 SmokeInterferenceSimulator 类 ---
class SmokeInterferenceSimulator:
    """
    烟幕干扰模拟器。
    包含轨迹模拟、烟幕扩散、相交计算等核心逻辑。
    """
    def __init__(self):
        # 模拟参数
        self.time_step = 0.1 # 秒
        self.smoke_effective_radius = 10.0 # 烟幕有效半径 (米)
        self.smoke_expansion_rate = 50.0 # 烟幕扩散速率 (米/秒)
        self.ground_level = 0.0 # 地面高度 (米)

    def simulate_missile_trajectory(self, missile_info):
        """模拟导弹轨迹 (简化为匀速直线运动)"""
        traj = Trajectory()
        # --- 关键修改：确保初始位置是浮点数 ---
        start_pos = np.array(missile_info['position'], dtype=float)
        speed = missile_info['speed']
        # 假设导弹飞向原点 (0, 0, 0)
        direction_norm = np.linalg.norm(start_pos)
        if direction_norm > 1e-10:
            direction = -start_pos / direction_norm
        else:
            direction = np.array([0.0, 0.0, -1.0]) # 默认向下
        
        time = 0.0
        current_pos = start_pos.copy() # copy 保持 float 类型
        traj.add_point(time, current_pos)
        
        # 模拟直到接近原点或撞击地面
        while np.linalg.norm(current_pos) > 10 and current_pos[2] > self.ground_level:
            time += self.time_step
            displacement = speed * self.time_step * direction
            current_pos = current_pos + displacement # 使用 + 而不是 += 有时也能避免就地修改类型问题，但copy更可靠
            # 简单防止钻地
            if current_pos[2] < self.ground_level:
                 current_pos[2] = self.ground_level
            traj.add_point(time, current_pos)
            
        return traj

    def simulate_drone_trajectory(self, drone_info, flight_params):
        """模拟无人机轨迹 (简化为匀速直线运动)"""
        traj = Trajectory()
        # --- 关键修改：确保初始位置是浮点数 ---
        start_pos = np.array(drone_info['position'], dtype=float)
        speed = flight_params['speed']
        direction_angle = flight_params['direction'] # 方位角 (弧度)
        #在同一水平面飞行，高度不变
        direction_vector = np.array([np.cos(direction_angle), np.sin(direction_angle), 0.0])
        
        time = 0.0
        current_pos = start_pos.copy() # copy 保持 float 类型
        traj.add_point(time, current_pos)
        
        # 模拟一段时间，例如直到烟幕弹释放后一段时间
        # 这里简化为模拟 10 秒
        while time <= 10.0:
            time += self.time_step
            displacement = speed * self.time_step * direction_vector
            current_pos = current_pos + displacement # 使用 + 而不是 +=
            traj.add_point(time, current_pos)
            
        return traj

    # ... (simulate_smoke_bomb_trajectory, simulate_smoke_cloud, ray_sphere_intersection_time, line_sphere_intersection 方法保持不变) ...
    # 为了完整性，这里也包含它们，但主要修改在上面两个方法
    def simulate_smoke_bomb_trajectory(self, drone_traj, release_time, explode_delay):
        """模拟烟幕弹轨迹 (简化为自由落体)"""
        traj = Trajectory()
        release_pos = drone_traj.get_position_at_time(release_time)
        
        # 简化：烟幕弹释放时速度与无人机相同，之后只受重力影响
        # 重力加速度
        g = 9.8 # m/s^2
        # 计算水平初速度
        t_after_release = release_time + 0.1
        if t_after_release <= drone_traj.times[-1]:
            pos_after_release = drone_traj.get_position_at_time(t_after_release)
            initial_velocity_xy = (pos_after_release - release_pos) / 0.1
        else:
            # 如果0.1秒后超出轨迹，使用最后两点计算
            if len(drone_traj.times) >= 2 and drone_traj.times[-1] > drone_traj.times[-2]:
                dt = drone_traj.times[-1] - drone_traj.times[-2]
                if dt > 0:
                    initial_velocity_xy = (drone_traj.positions[-1] - drone_traj.positions[-2]) / dt
                else:
                    initial_velocity_xy = np.array([0.0, 0.0])
            else:
                initial_velocity_xy = np.array([0.0, 0.0])
        # inital_velocity_xy = drone_info['direction']
        initial_velocity_z = 0.0 # 假设垂直初速度为0
        
        time = release_time
        current_pos = np.array(release_pos, dtype=float) # 确保是 float
        current_velocity_z = initial_velocity_z
        
        traj.add_point(time, current_pos)
        
        explode_time = release_time + explode_delay
        while time < explode_time:
            time += self.time_step
            # 水平方向匀速
            delta_xy = initial_velocity_xy * self.time_step
            # 垂直方向匀加速
            delta_z = current_velocity_z * self.time_step - 0.5 * g * self.time_step**2
            
            current_pos[0] += delta_xy[0]
            current_pos[1] += delta_xy[1]
            current_pos[2] += delta_z
            
            current_velocity_z -= g * self.time_step
            
            # 防止钻地
            if current_pos[2] < self.ground_level:
                current_pos[2] = self.ground_level
                # 在落地点添加轨迹点
                traj.add_point(time, current_pos)
                break # 落地后停止
            
            traj.add_point(time, current_pos)
            
        # 确保爆炸时间点被精确记录 (如果未因落地而提前终止)
        if abs(time - explode_time) > 1e-6 and current_pos[2] >= self.ground_level - 1e-6: # Fallback check
             # 重新计算爆炸时刻的位置
             dt_explode = explode_time - release_time
             final_pos = release_pos + initial_velocity_xy * dt_explode + np.array([0, 0, initial_velocity_z * dt_explode - 0.5 * g * dt_explode**2])
             traj.add_point(explode_time, final_pos)
        
        return traj, explode_time

    def simulate_smoke_cloud(self, explode_pos, explode_time, duration=100.0):
        """模拟烟幕云团扩散 (简化为随时间增大半径的球体)"""
        traj = Trajectory()
        center = np.array(explode_pos, dtype=float) # 确保是 float
        
        # 简化：烟幕中心位置不变，只记录不同时刻的半径
        # 这里用位置来表示中心，用时间来关联半径
        time = explode_time
        end_time = explode_time + duration
        while time <= end_time:
            # 半径信息隐含在时间中，对于这个 Trajectory 类，我们只记录中心
            traj.add_point(time, center) 
            time += self.time_step
            
        return traj

    def ray_sphere_intersection_time(self, missile_trajectory, smoke_center, smoke_radius):
        """计算导弹轨迹与烟幕球的相交时间段"""
        intersection_intervals = []
        
        positions = missile_trajectory.get_positions_array()
        times = np.array(missile_trajectory.times)
        
        # --- 添加检查 ---
        if len(positions) < 2:
            print("Warning: Missile trajectory has less than 2 points, cannot compute intersection.")
            return intersection_intervals

        for i in range(len(positions) - 1):
            # 当前时间段的射线
            p1, p2 = positions[i], positions[i + 1]
            t1, t2 = times[i], times[i + 1]
            
            # 射线方向
            ray_dir_vec = p2 - p1
            ray_dir_norm = np.linalg.norm(ray_dir_vec)
            if ray_dir_norm > 1e-10: # 避免除以零
                ray_dir = ray_dir_vec / ray_dir_norm
            else:
                continue # 跳过静止点或重合点
            
            # 检查与球的相交
            intersect_params = self.line_sphere_intersection(p1, ray_dir, smoke_center, smoke_radius)
            
            if intersect_params:
                t_start, t_end = intersect_params
                # 转换到实际时间 (注意：这里的 t_start, t_end 是相对于 p1 点的距离参数)
                # p = p1 + t * (p2-p1) => t_param = t / |p2-p1|
                # 所以实际时间是 t1 + t_param * (t2-t1)
                segment_duration = t2 - t1
                if segment_duration > 0:
                     actual_t_start = t1 + (t_start / ray_dir_norm) * segment_duration
                     actual_t_end = t1 + (t_end / ray_dir_norm) * segment_duration
                     
                     # 限制在当前时间段内
                     actual_t_start = max(actual_t_start, t1)
                     actual_t_end = min(actual_t_end, t2)
                     
                     if actual_t_start < actual_t_end:
                         intersection_intervals.append((actual_t_start, actual_t_end))
        
        # 合并重叠的时间段
        if not intersection_intervals:
            return []
        
        intersection_intervals.sort()
        merged_intervals = [intersection_intervals[0]]
        for current in intersection_intervals[1:]:
            last = merged_intervals[-1]
            if current[0] <= last[1]: # 重叠或相邻
                merged_intervals[-1] = (last[0], max(last[1], current[1]))
            else:
                merged_intervals.append(current)
        return merged_intervals

    def line_sphere_intersection(self, ray_start, ray_dir, sphere_center, sphere_radius):
        """射线与球面相交计算"""
        # 射线方程: P = ray_start + t * ray_dir (t >= 0)
        # 球面方程: |P - sphere_center|² = sphere_radius²
        
        oc = ray_start - sphere_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - sphere_radius**2
        
        # 避免除以零
        if abs(a) < 1e-10:
             # 射线方向为零向量，检查起点是否在球内
             if c <= 0:
                  return (0, np.inf) # 整个射线都在球内（虽然方向为0）
             else:
                  return None # 无交点
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None  # 无交点
        
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2*a)
        t2 = (-b + sqrt_discriminant) / (2*a)
        
        # 只考虑前向射线 (t >= 0)
        if t2 < 0:
            return None
        
        t1 = max(0, t1)  # 确保时间为正
        
        return (t1, t2)

# --- 3. 修改后的 TrajectoryVisualizer 类 ---
class TrajectoryVisualizer:
    def __init__(self, figsize=(12, 9)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def plot_trajectory(self, trajectory, label='Trajectory', color='blue', style='-'):
        """绘制轨迹"""
        if not trajectory.times:
             print(f"Warning: Trajectory '{label}' is empty.")
             return
        positions = trajectory.get_positions_array()
        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    color=color, linestyle=style, label=label, linewidth=2)
    
    def plot_smoke_cloud(self, center, radius, alpha=0.3, color='red'):
        """绘制烟幕云团"""
        # 创建球面
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        self.ax.plot_surface(x, y, z, alpha=alpha, color=color)
    
    def plot_target(self, position, radius=7, height=10, color='green'):
        """绘制圆柱形目标"""
        # 圆柱底面
        theta = np.linspace(0, 2*np.pi, 50)
        x_base = radius * np.cos(theta) + position[0]
        y_base = radius * np.sin(theta) + position[1]
        z_base = np.full_like(x_base, position[2])
        
        # 圆柱顶面
        z_top = np.full_like(x_base, position[2] + height)
        
        # 绘制底面和顶面
        self.ax.plot(x_base, y_base, z_base, color=color, linewidth=2)
        self.ax.plot(x_base, y_base, z_top, color=color, linewidth=2)
        
        # 绘制侧面线条
        for i in range(0, len(theta), 5):
            self.ax.plot([x_base[i], x_base[i]], [y_base[i], y_base[i]], 
                        [z_base[i], z_top[i]], color=color, alpha=0.6)
    
    def setup_scene(self, xlim=(-5000, 25000), ylim=(-5000, 5000), zlim=(0, 2500)):
        """设置场景"""
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 避免重复图例
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        
        self.ax.grid(True)
        
    def show(self):
        plt.show()

# --- 4. 修改后的 AnimatedTrajectoryVisualizer 类 ---
class AnimatedTrajectoryVisualizer(TrajectoryVisualizer): # 继承
    def __init__(self, figsize=(12, 9)):
        super().__init__(figsize) # 调用父类初始化
        self.trajectories = {}
        self.smoke_clouds = [] # 需要额外信息来动画化烟幕
        
    def add_trajectory(self, name, trajectory, color='blue', style='-'):
        self.trajectories[name] = {
            'trajectory': trajectory,
            'color': color,
            'style': style,
            'line': None
        }
    
    def animate(self, frame):
        self.ax.clear() # 清除上一帧
        
        # 计算当前时间
        if not self.trajectories:
            return
        max_time = max([max(traj['trajectory'].times) for traj in self.trajectories.values() if traj['trajectory'].times])
        if not max_time:
             return
        current_time = frame * max_time / 200  # 200帧动画
        
        # 绘制轨迹（到当前时间为止）
        for name, traj_info in self.trajectories.items():
            trajectory = traj_info['trajectory']
            if not trajectory.times:
                 continue
            positions = []
            times = trajectory.times
            
            for i, t in enumerate(times):
                if t <= current_time:
                    positions.append(trajectory.positions[i])
                else:
                    break
            
            if len(positions) > 1:
                positions = np.array(positions)
                self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                           color=traj_info['color'], linestyle=traj_info['style'], 
                           label=name, linewidth=2)
                
                # 绘制当前位置点
                current_pos = trajectory.get_position_at_time(current_time)
                self.ax.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]],
                              color=traj_info['color'], s=50, label=f"{name} (Current)" if frame==0 else "") # 避免图例重复
        
        # 简化：动画中不绘制烟幕云团，因为缺少详细信息
        # 如果需要，可以存储烟幕的中心和随时间变化的半径信息
        
        self.setup_scene() # 调用父类方法设置场景
        
    def create_animation(self, save_path=None):
         if not self.trajectories:
              print("No trajectories to animate.")
              return None
         ani = animation.FuncAnimation(self.fig, self.animate, frames=200, 
                                    interval=50, blit=False, repeat=True) # 添加 repeat=True
        
         if save_path:
            try:
                ani.save(save_path, writer='pillow', fps=20) # 提高 fps
                print(f"Animation saved to {save_path}")
            except Exception as e:
                 print(f"Failed to save animation: {e}")
        
         return ani

class missile:
    def __init__(self,missile_info,speed,position):
        self.position = position
        self.speed = 300
        self.missile_info['position'] = position
        self.missile_info['speed'] = speed
    
class drone:
    def __init__(self, position, speed, direction):
        # 检查飞行速度是否在70到140之间
        if not isinstance(speed, (int, float)):
            raise TypeError("速度必须是数字类型")
        if speed < 70 or speed > 140:
            raise ValueError(f"无人机飞行速度必须在70到140之间，当前速度: {speed}")
        
        self.position = position
        self.speed = speed
        self.direction = direction
        
    
# --- 5. 修改后的 solve_problem_1 函数 ---
def solve_problem_1():
    """问题1：FY1投放1枚烟幕弹对M1干扰"""
    simulator = SmokeInterferenceSimulator() # 实例化补充的类
    
    # 导弹M1信息
    missile_m1 = {
        'position': [20000, 0, 2000],
        'speed': 300
    }
    
    # 无人机FY1信息
    drone_fy1 = {
        'position': [17800, 0, 1800]
    }
    
    # 飞行参数
    flight_params = {
        'direction': np.pi,  # 朝向假目标（180度）
        'speed': 120
    }
    
    # 模拟轨迹
    print("Simulating missile trajectory...")
    missile_traj = simulator.simulate_missile_trajectory(missile_m1)
    print("Simulating drone trajectory...")
    drone_traj = simulator.simulate_drone_trajectory(drone_fy1, flight_params)
    
    # 烟幕弹轨迹
    release_time = 1.5
    explode_delay = 3.6
    print("Simulating smoke bomb trajectory...")
    smoke_traj, explode_time = simulator.simulate_smoke_bomb_trajectory(
        drone_traj, release_time, explode_delay
    )
    
    # 获取爆炸位置
    explode_pos = smoke_traj.get_position_at_time(explode_time)
    print(f"Smoke bomb explodes at position {explode_pos} at time {explode_time:.2f}s")

    # 模拟烟幕云团 (这里简化处理，只记录中心和爆炸时间)
    print("Simulating smoke cloud expansion...")
    cloud_traj = simulator.simulate_smoke_cloud(explode_pos, explode_time)
    
    # --- 计算遮蔽时间 (使用修改后的交集函数) ---
    print("Calculating intersection times...")
    # 假设烟幕有效半径为 simulator.smoke_effective_radius
    intersection_intervals = simulator.ray_sphere_intersection_time(
        missile_traj, explode_pos, simulator.smoke_effective_radius
    )
    
    total_coverage = 0.0
    if intersection_intervals:
        print("Missile passes through smoke cloud.")
        print("Intersection intervals (start, end):")
        for start, end in intersection_intervals:
             duration = end - start
             total_coverage += duration
             print(f"  ({start:.2f}s, {end:.2f}s), Duration: {duration:.2f}s")
        print(f"Total effective coverage time: {total_coverage:.2f} seconds")
    else:
        print("Missile does not pass through the smoke cloud.")
        print("Total effective coverage time: 0.00 seconds")

    
    # 可视化
    print("Generating visualization...")
    vis = TrajectoryVisualizer()
    vis.plot_trajectory(missile_traj, 'Missile M1', 'red', '-')
    vis.plot_trajectory(drone_traj, 'Drone FY1', 'blue', '-')
    vis.plot_trajectory(smoke_traj, 'Smoke Bomb', 'orange', '--')
    # cloud_traj 在这个简化模型中只记录了中心点，绘制它意义不大
    # vis.plot_trajectory(cloud_traj, 'Smoke Cloud Center', 'gray', ':') 
    
    # 绘制目标 (真目标在原点)
    vis.plot_target([0, 0, 0], radius=50, height=50, color='green') # 调整目标大小以便观察
    # 绘制爆炸时的烟幕球
    vis.plot_smoke_cloud(explode_pos, simulator.smoke_effective_radius, alpha=0.2, color='red')
    
    vis.setup_scene(xlim=(-5000, 25000), ylim=(-5000, 5000), zlim=(0, 2500))
    vis.show()
    
    # 动画可视化
    print("Generating animation...")
    anim_vis = AnimatedTrajectoryVisualizer()
    anim_vis.add_trajectory('Missile M1', missile_traj, 'red', '-')
    anim_vis.add_trajectory('Drone FY1', drone_traj, 'blue', '-')
    anim_vis.add_trajectory('Smoke Bomb', smoke_traj, 'orange', '--')
    # anim_vis.add_trajectory('Smoke Cloud', cloud_traj, 'gray', ':')
    ani = anim_vis.create_animation() # 不保存文件，直接在 notebook/Jupyter 中显示
    if ani:
        from IPython.display import HTML # 如果在 Jupyter Notebook 中运行
        # display(HTML(ani.to_jshtml())) # 显示动画 (取消注释以在 Jupyter 中使用)
        plt.show() # 显示最后一帧或静态图
    else:
        plt.show() # 显示静态图
        
    return total_coverage

# --- 6. 运行示例 ---
if __name__ == "__main__":
    print("--- Running Smoke Interference Simulation ---")
    coverage_time = solve_problem_1()
    print(f"\n--- Simulation Complete ---")
    print(f"Final reported coverage time: {coverage_time:.2f} seconds")





