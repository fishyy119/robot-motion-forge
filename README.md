# Robot Motion Forge

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![MuJoCo 3.17](https://img.shields.io/badge/MuJoCo-3.17-00897B?style=flat)](https://mujoco.org/)
[![PyCeres 2.6](https://img.shields.io/badge/PyCeres-2.6-546E7A?style=flat)](https://github.com/cvg/pyceres)

手动关键帧 + 整段轨迹优化 创作机器人动作

## 定义任务机器人

以 [机器人配置样例](configs/robots/robot.example.json) 为起点创建本机配置，编辑器加载本地的 MuJoCo MJCF。MJCF 使用只有一个根连杆的开链结构，平地由编辑器添加，模型文件只包含机器人。

- `robot:` 机器人模型。
  - `name:` 编辑器中显示的机器人名称。
  - `repository_root:` 机器人仓库根目录。相对路径以配置文件所在目录为基准。
  - `model:` 机器人 MJCF 路径。相对路径以机器人仓库根目录为基准。
  - `joint_limit_overrides:` 按关节名覆盖模型的位置限位。
- `home:` 新工程的初始姿态。省略的值沿用模型初始状态。
  - `position_m:` 浮动基座的初始位置，单位为米。
  - `quaternion_wxyz:` 浮动基座的初始朝向，四元数按 `wxyz` 顺序填写。
  - `joints_rad:` 旋转关节的初始角度，按关节名填写，单位为弧度。
  - `joints_m:` 直线关节的初始位置，按关节名填写，单位为米。
- `frames:` 头、手、脚等部位的控制坐标系。
  - `<坐标系名称>:` 坐标系的键，例如 `head` 或 `left_foot`。
    - `body:` 附着连杆的名称，坐标轴与该连杆对齐。
    - `offset_m:` 原点在连杆局部坐标中的偏移，单位为米。
    - `support_polygon_cm:` 用于足部支撑的局部足底轮廓，顶点单位为厘米。
    - `control_mode:` 拖拽时的默认控制分量。
    - `fixed_mode:` 新会话中的默认固定分量。
    - `trajectory_mode:` 新工程中的默认轨迹跟踪分量。
    - `trajectory_reference_body:` 参考连杆，配合 `trajectory_mode` 使用。省略时使用世界坐标系。
- `joint_groups:` 按模型中的关节名分组，用于姿态面板和关节曲线。

完整字段定义见 [机器人配置 Schema](configs/robots/robot-definition.schema.json)。

## 保存与打开工程

使用如下命令启动程序，

```shell
robot-motion-forge # 采用默认机器人 configs/robots/robot.json 启动
robot-motion-forge --robot <path-to-robot.json> --project <name-of-project.json>  # 指定机器人与加载工程
```

程序固定将工程保存在本地机器人资产目录 `<robot.repository_root>/.robot-motion-forge/`，不支持其他路径。

程序生命周期中机器人模型固定，工程可以在在「工程」页进行加载、保存、替换等操作。

## 编辑关键帧

在时间轴定位到关键帧后，可通过拖拽末端控制点或各关节调整关键帧姿态。右键各末端控制点可以调整固定模式与控制模式，右键各关节可是单独锁定各关节，避免拖动末端控制点时因自由度过高导致 IK 优化出不理想的姿态。

每张关键帧有两个独立的求解选项。

| 选项 | 作用 |
| --- | --- |
| **锁定姿态** | 固定该时刻的基座位姿和关节位置。 |
| **速度归零** | 将该时刻的基座线速度、角速度和各关节速度固定为零。 |

## 轨迹优化

以各采样节点的姿态与广义速度为优化变量，使用 PyCeres 将整段动作组织成一个稀疏非线性最小二乘问题进行优化。根据关键帧进行逐关节插值（PCHIP）生成参考轨迹，以此用作优化求解的初值（考虑到末端点跟踪目标与逐关节插值参考相悖，因此参考轨迹仅用于设定初值）。

### 终端条件

- **静止。** 首尾速度固定为零。
- **循环。** 首尾共用一个求解节点，姿态和速度在接缝处相同。
- **平移循环。** 首尾的基座水平位置分别求解，基座高度、朝向、关节位置和速度在接缝处相同。

### 求解约束

| 软约束与代价 | 作用 |
| --- | --- |
| 未锁定帧权重 | 在关键帧未锁定时，惩罚求解姿态与编辑姿态的偏差。 |
| 速度正则、加速度平滑 | 对整段轨迹的速度和加速度施加代价。 |
| 坐标系轨迹 | 按所选模式跟踪该部位的位置、朝向或完整位姿。每项可选择参考坐标系和相对权重。 |
| 自碰撞、地面碰撞 | 惩罚自碰撞间隙不足和地面穿透。 |
| 足部支撑 | 惩罚支撑足的贴地与水平误差，以及质心投影的支撑裕量不足。该项采用准静态判据，**不适用**于步行等动态动作。 |

以上约束可以通过将权重设为 0 关闭。
