import numpy as np

DEFAULT_RADIUS = 0.6
AXIS_LEN = 2.2
AXIS_COLORS = {
    "x": [235, 60, 60],
    "y": [60, 200, 90],
    "z": [70, 130, 240],
}


def rot_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def body_axes(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_matrix(roll, pitch, yaw).T


def arc_points(center, axis_from, axis_to, angle, radius=1.3, n=24):
    center = np.asarray(center, float)
    u = np.asarray(axis_from, float)
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.asarray(axis_to, float)
    w = w - u * (w @ u)
    w = w / (np.linalg.norm(w) + 1e-12)
    ts = np.linspace(0.0, angle, n)
    return center + radius * (np.cos(ts)[:, None] * u + np.sin(ts)[:, None] * w)


def demo_data_6d() -> dict:
    n = 49
    obs = set()
    for i in range(0, n, 2):
        obs |= {(i, 0, 0), (i, n - 1, 0), (0, i, 0), (n - 1, i, 0)}
    edge = [(1, 15), (1, 31), (47, 16), (47, 32), (16, 1), (32, 1), (15, 47), (31, 47)]
    obs |= {(x, y, 0) for x, y in edge}
    obs = sorted(obs)

    def make(start, goal, z_peak, T=34.0, steps=200):
        ts = np.linspace(0, T, steps)
        s = (1 - np.cos(np.pi * ts / T)) / 2
        mid = (np.array(start) + np.array(goal)) / 2 + np.array([4, -4])
        p = np.array([(1 - u) ** 2 * np.array(start) + 2 * (1 - u) * u * mid + u ** 2 * np.array(goal) for u in s])
        z = z_peak * np.sin(np.pi * s)
        xyz = np.column_stack([p, z])
        d = np.gradient(xyz, axis=0)
        speed = np.linalg.norm(d, axis=1) + 1e-9
        yaw = np.arctan2(d[:, 1], d[:, 0])
        pitch = np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1))
        yaw_rate = np.gradient(np.unwrap(yaw))
        roll = np.clip(-4.0 * yaw_rate / speed, -0.6, 0.6)
        traj = np.column_stack([ts, xyz, roll, pitch, yaw])
        planned = xyz[::5]
        return traj.tolist(), planned.tolist()

    agents = []
    for name, col, st, gl, zp in [
        ("red", [230, 50, 40], (4, 4), (44, 44), 0.0),
        ("blue", [30, 60, 240], (44, 44), (4, 4), 6.0),
        ("magenta", [230, 50, 230], (6, 42), (39, 12), 0.0),
        ("green", [40, 180, 80], (39, 12), (6, 42), 6.0),
    ]:
        tr, pl = make(st, gl, zp)
        agents.append({"name": name, "color": col, "radius": DEFAULT_RADIUS, "traj": tr, "planned": pl})
    return {"cell_size": 1.0, "obstacle_size": 0.6, "obstacles": [list(o) for o in obs], "agents": agents}