from isaacsim import SimulationApp

# 必须最先创建 SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

# 启用 ROS2 Bridge 扩展
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

# 更新几帧，让扩展完成加载
for _ in range(10):
    simulation_app.update()


import numpy as np
import omni.graph.core as og
import omni.timeline
import omni.usd
from pxr import UsdPhysics

from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka

def set_franka_drive_to_effort_mode(
    franka_prim_path: str = "/World/Franka",
    stiffness: float = 0.0,
    damping: float = 0.0,
    max_force: float = 1000.0,
):
    """
    将 Franka 关节 drive 改成更接近 effort / torque 控制的模式。

    stiffness = 0:
        关闭位置弹簧，不再主动把关节拉回 target position。

    damping = 0 或很小:
        关闭或减小速度阻尼。

    max_force:
        drive 最大力矩限制。这里保留较大值，避免 drive API 限制。
    """
    stage = omni.usd.get_context().get_stage()

    joint_count = 0
    angular_drive_count = 0
    linear_drive_count = 0

    print("Setting Franka joint drives...")
    print(f"Target robot prim: {franka_prim_path}")

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())

        if not prim_path.startswith(franka_prim_path):
            continue

        type_name = prim.GetTypeName()

        # Franka 的关节一般是 PhysicsRevoluteJoint
        if "Joint" not in type_name:
            continue

        joint_count += 1
        print(f"[Joint] {prim_path}, type={type_name}")

        # 旋转关节：angular drive
        try:
            angular_drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            angular_drive.CreateStiffnessAttr().Set(float(stiffness))
            angular_drive.CreateDampingAttr().Set(float(damping))
            angular_drive.CreateMaxForceAttr().Set(float(max_force))

            print(
                f"  angular drive set: stiffness={stiffness}, "
                f"damping={damping}, max_force={max_force}"
            )
            angular_drive_count += 1
        except Exception as e:
            print(f"  angular drive skipped: {e}")

        # 直线关节：linear drive，Franka 主体一般用不到，但保留兼容
        try:
            linear_drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            linear_drive.CreateStiffnessAttr().Set(float(stiffness))
            linear_drive.CreateDampingAttr().Set(float(damping))
            linear_drive.CreateMaxForceAttr().Set(float(max_force))

            print(
                f"  linear drive set: stiffness={stiffness}, "
                f"damping={damping}, max_force={max_force}"
            )
            linear_drive_count += 1
        except Exception:
            pass

    print(f"Total joint prims found: {joint_count}")
    print(f"Angular drives modified: {angular_drive_count}")
    print(f"Linear drives modified: {linear_drive_count}")
def create_ros2_action_graph(franka_prim_path: str):
    """
    创建 ROS2 控制图：

    Isaac Sim 发布：
        /joint_states

    Isaac Sim 订阅：
        /joint_command

    外部 ROS2 可以通过 /joint_command 控制 Franka。
    """
    graph_path = "/World/ROS2Graph"

    print("Creating ROS2 Action Graph...")

    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],

            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),

                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),

                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],

            og.Controller.Keys.SET_VALUES: [
                # topic 名字不要加 /，官方示例也是 joint_states / joint_command
                ("PublishJointState.inputs:topicName", "joint_states"),
                ("SubscribeJointState.inputs:topicName", "joint_command"),

                # 关键修改 1：PublishJointState 直接设置 targetPrim 为字符串路径
                ("PublishJointState.inputs:targetPrim", franka_prim_path),

                # 关键修改 2：ArticulationController 用 robotPath
                # 5.x 版本里 robotPath 优先级更明确
                ("ArticulationController.inputs:robotPath", franka_prim_path),
            ],
        },
    )

    print("ROS2 Action Graph created.")
    print(f"Robot path: {franka_prim_path}")
    print("Publish topic: /joint_states")
    print("Subscribe topic: /joint_command")
def main():
    # 创建仿真世界
    world = World(stage_units_in_meters=1.0)

    # 添加地面
    world.scene.add_default_ground_plane()

    # 添加 Franka Panda
    franka = world.scene.add(
        Franka(
            prim_path="/World/Franka",
            name="franka",
            position=np.array([0.0, 0.0, 0.0]),
        )
    )

    # 关键：关闭 Franka 关节 position drive
    # 这样 /joint_command.effort 才更接近纯力矩控制
    set_franka_drive_to_effort_mode(
        franka_prim_path="/World/Franka",
        stiffness=20.0,
        damping=5.0,
        max_force=1000.0,
    )
    # 初始化仿真
    world.reset()

    print("Franka loaded successfully.")
    print("Prim path:", franka.prim_path)

    # 创建 ROS2 控制图
    create_ros2_action_graph(franka.prim_path)

    # 启动 timeline，相当于点击 Isaac Sim 里的 Play
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    print("Simulation started.")
    print("Now you can open another terminal and run:")
    print("  source /opt/ros/humble/setup.bash")
    print("  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp")
    print("  ros2 topic list")
    print("You should see:")
    print("  /joint_states")
    print("  /joint_command")

    # 保持 Isaac Sim GUI 运行
    while simulation_app.is_running():
        world.step(render=True)

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()