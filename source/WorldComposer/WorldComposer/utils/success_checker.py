import numpy as np
import torch


def step_interval(interval=50):
    """Create a decorator that evaluates every fixed number of calls."""

    def decorator(func):
        call_count = 0

        def wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count % interval == 0:
                return func(*args, **kwargs)
            else:
                return False

        return wrapper

    return decorator


def calculate_distance(point_a, point_b):
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)


def get_object_particle_position(particle_object, index_list):
    position = (
        particle_object._cloth_prim_view.get_world_positions()
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        * 100
    )
    select_position = []
    for index in index_list:
        select_position.append(tuple(position[index]))
    return select_position

@step_interval(interval=50)
def success_checker_orangeinbowl(
    bowl_object_a, bowl_object_b
):
    position_a = bowl_object_a._position()
    position_b = bowl_object_b._position()
    a_x = position_a[0].item()
    a_y = position_a[1].item()
    a_z = position_a[2].item()
    b_x = position_b[0].item()
    b_y = position_b[1].item()
    b_z = position_b[2].item()
    print(calculate_distance(a_x, b_x))
    print(calculate_distance(a_y, b_y))
    print(calculate_distance(a_z, b_z))
    success = (
        calculate_distance(a_x, b_x) <= 0.05
        and calculate_distance(a_y, b_y) <= 0.05
        and calculate_distance(a_z, b_z) <= 0.05
    )
    return bool(success)

# @step_interval(interval=50)
def success_checker_pour(
    fluid_system,
    bowl_pos: np.ndarray,
    sample_count: int = 100,
    xy_tolerance: float = 0.08,
    z_tolerance: float = 0.12,
    success_ratio: float = 0.5,
):
    """Check whether enough sampled fluid particles fall near the container."""
    # Read positions via the FluidSystem helper (does not depend on updateToUsd)
    particles = fluid_system.get_particle_positions_world()
    if len(particles) == 0:
        return False

    sample_count = min(sample_count, len(particles))
    indices = np.random.choice(len(particles), size=sample_count, replace=False)
    sampled = particles[indices]

    container_pos = np.array(bowl_pos, dtype=np.float32)

    diff = np.abs(sampled - container_pos)
    in_xy = (diff[:, 0] <= xy_tolerance) & (diff[:, 1] <= xy_tolerance)
    in_z = diff[:, 2] <= z_tolerance
    inside = in_xy & in_z

    ratio = inside.mean() if len(inside) > 0 else 0.0
    return bool(ratio >= success_ratio)
    
@step_interval(interval=50)
def success_checker_bowlinplate(
    rigid_object_a, rigid_object_b, env_id: int = 0
):
    pos_a = rigid_object_a.data.root_pos_w[env_id]      # (3,)
    pos_b = rigid_object_b.data.root_pos_w[env_id]
    a_x = pos_a[0].item()
    a_y = pos_a[1].item()
    b_x = pos_b[0].item()
    b_y = pos_b[1].item()

    print(calculate_distance(a_x, b_x))
    print(calculate_distance(a_y, b_y))
    success = (
        calculate_distance(a_x, b_x) <= 0.035
        and calculate_distance(a_y, b_y) <= 0.035
    )
    return bool(success)


@step_interval(interval=50)
def success_checker_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)
    success = (
        calculate_distance(p[0], p[4]) <= 10
        and calculate_distance(p[2], p[3]) <= 16
        and calculate_distance(p[1], p[5]) <= 10
    )
    return bool(success)


@step_interval(interval=50)
def success_checker_AinB(
    rigid_object_a, rigid_object_b, env_id: int = 0, xy_threshold: float = 0.03
):
    """Generic success check based on the planar distance between rigid bodies A and B."""
    pos_a = rigid_object_a.data.root_pos_w[env_id]
    pos_b = rigid_object_b.data.root_pos_w[env_id]

    a_x = pos_a[0].item()
    a_y = pos_a[1].item()
    b_x = pos_b[0].item()
    b_y = pos_b[1].item()

    print(calculate_distance(a_x, b_x))
    print(calculate_distance(a_y, b_y))

    success = (
        calculate_distance(a_x, b_x) <= xy_threshold
        and calculate_distance(a_y, b_y) <= xy_threshold
    )
    return bool(success)











@step_interval(interval=50)
def success_checker_fling(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)

    def xy_distance(a, b):
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

    def z_distance(a, b):
        return abs(a[2] - b[2])

    success = (
        xy_distance(p[0], p[4]) > 18
        and z_distance(p[0], p[4]) < 2
        and xy_distance(p[1], p[5]) > 18
        and z_distance(p[1], p[5]) < 2
    )

    return bool(success)


@step_interval(interval=30)
def success_checker_burger(beef_pos, plate_pos):
    diff_xy = beef_pos[:, :2] - plate_pos[:, :2]
    dist_xy = torch.linalg.norm(diff_xy, dim=-1)

    diff_z = torch.abs(beef_pos[:, 2] - plate_pos[:, 2])

    success_mask = (dist_xy < 0.045) & (diff_z < 0.03)
    success = success_mask.any().item()

    return bool(success)


@step_interval(interval=30)
def success_checker_rubbish(food_rubbish01_pos, food_rubbish02_pos, food_rubbish03_pos, desktop_dustpan_pos, dustpan_size_x=0.15, dustpan_size_y=0.15):

    diff_xy_1 = food_rubbish01_pos[:, :2] - desktop_dustpan_pos[:, :2]
    dist_x_1 = torch.abs(diff_xy_1[:, 0])
    dist_y_1 = torch.abs(diff_xy_1[:, 1])
    in_dustpan_1 = (dist_x_1 <= dustpan_size_x) & (dist_y_1 <= dustpan_size_y)

    diff_xy_2 = food_rubbish02_pos[:, :2] - desktop_dustpan_pos[:, :2]
    dist_x_2 = torch.abs(diff_xy_2[:, 0])
    dist_y_2 = torch.abs(diff_xy_2[:, 1])
    in_dustpan_2 = (dist_x_2 <= dustpan_size_x) & (dist_y_2 <= dustpan_size_y)
    
    diff_xy_3 = food_rubbish03_pos[:, :2] - desktop_dustpan_pos[:, :2]
    dist_x_3 = torch.abs(diff_xy_3[:, 0])
    dist_y_3 = torch.abs(diff_xy_3[:, 1])
    in_dustpan_3 = (dist_x_3 <= dustpan_size_x) & (dist_y_3 <= dustpan_size_y)
    
    success_mask = in_dustpan_1 & in_dustpan_2 & in_dustpan_3
    success = success_mask.any().item()
    
    return bool(success)

@step_interval(interval=6)
def success_checker_cut(sausage_count: int) -> bool:
    return sausage_count >= 2
