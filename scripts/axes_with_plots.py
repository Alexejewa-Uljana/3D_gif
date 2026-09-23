import argparse
import json

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import orient_common as oc

ANGLE_COLORS = {"roll": [235, 60, 60], "pitch": [60, 200, 90], "yaw": [70, 130, 240]}


def set_time(t):
    rr.set_time("sim_time", duration=t) if hasattr(rr, "set_time") else rr.set_time_seconds("sim_time", t)


def log_scalar(path, value):
    rr.log(path, rr.Scalars(value) if hasattr(rr, "Scalars") else rr.Scalar(value))


def build_blueprint(names):
    plots = [rrb.TimeSeriesView(origin=f"plots/{n}", name=f"{n} agent's degrees") for n in names]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="scene"),
            rrb.Grid(contents=plots),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )


def log_scene(data):
    cs = float(data.get("cell_size", 1.0))
    obs = np.asarray(data.get("obstacles", []), float).reshape(-1, 3)
    if len(obs):
        half = float(data.get("obstacle_size", 0.6)) * cs / 2
        c = obs * cs
        c[:, 2] += half
        rr.log("world/obstacles",
               rr.Boxes3D(centers=c, half_sizes=np.full((len(obs), 3), half),
                          colors=[140, 190, 140], fill_mode="solid"), static=True)

    for ag in data["agents"]:
        name, color = ag["name"], ag.get("color", [100, 100, 255])
        r = float(ag.get("radius", oc.DEFAULT_RADIUS))
        traj = np.asarray(ag["traj"], float)
        if "planned" in ag:
            rr.log(f"world/planned/{name}",
                   rr.LineStrips3D([np.asarray(ag["planned"], float) * cs], colors=[240, 170, 40], radii=0.05),
                   static=True)
        rr.log(f"world/agents/{name}/body",
               rr.Ellipsoids3D(half_sizes=[[r, r, r]], colors=[color], fill_mode="solid"), static=True)
        rr.log(f"world/agents/{name}/triad",
               rr.Arrows3D(origins=[[0, 0, 0]] * 3,
                           vectors=[[oc.AXIS_LEN, 0, 0], [0, oc.AXIS_LEN, 0], [0, 0, oc.AXIS_LEN]],
                           colors=[oc.AXIS_COLORS["x"], oc.AXIS_COLORS["y"], oc.AXIS_COLORS["z"]],
                           radii=0.06), static=True)
        for ang, col in ANGLE_COLORS.items():
            rr.log(f"plots/{name}/{ang}", rr.SeriesLines(colors=[col], names=[ang]), static=True)

        for i, row in enumerate(traj):
            t, x, y, z, roll, pitch, yaw = row
            set_time(t)
            rr.log(f"world/agents/{name}",
                   rr.Transform3D(translation=[x * cs, y * cs, z * cs + r], mat3x3=oc.rot_matrix(roll, pitch, yaw)))
            rr.log(f"world/trails/{name}",
                   rr.LineStrips3D([traj[: i + 1, 1:4] * cs], colors=[color], radii=0.08))
            log_scalar(f"plots/{name}/roll", np.degrees(roll))
            log_scalar(f"plots/{name}/pitch", np.degrees(pitch))
            log_scalar(f"plots/{name}/yaw", np.degrees(yaw))


def main():
    ap = argparse.ArgumentParser(description="D: направления без подписей + 4 графика углов")
    ap.add_argument("json", nargs="?")
    ap.add_argument("--save", metavar="FILE.rrd")
    a = ap.parse_args()
    data = json.load(open(a.json, encoding="utf-8")) if a.json else oc.demo_data_6d()
    names = [ag["name"] for ag in data["agents"]]

    rr.init("mapf_orient_d", spawn=not a.save)
    if a.save:
        rr.save(a.save)
    rr.send_blueprint(build_blueprint(names))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_scene(data)


if __name__ == "__main__":
    main()