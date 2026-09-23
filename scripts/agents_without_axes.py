import argparse
import json

import numpy as np
import rerun as rr

import orient_common as oc


def set_time(t):
    rr.set_time("sim_time", duration=t) if hasattr(rr, "set_time") else rr.set_time_seconds("sim_time", t)


def log_obstacles(data):
    cs = float(data.get("cell_size", 1.0))
    obs = np.asarray(data.get("obstacles", []), float).reshape(-1, 3)
    if len(obs):
        half = float(data.get("obstacle_size", 0.6)) * cs / 2
        c = obs * cs
        c[:, 2] += half
        rr.log("world/obstacles",
               rr.Boxes3D(centers=c, half_sizes=np.full((len(obs), 3), half),
                          colors=[140, 190, 140], fill_mode="solid"), static=True)


def log_scene(data):
    cs = float(data.get("cell_size", 1.0))
    log_obstacles(data)
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
        for i, row in enumerate(traj):
            t, x, y, z = row[0], row[1], row[2], row[3]
            set_time(t)
            rr.log(f"world/agents/{name}", rr.Transform3D(translation=[x * cs, y * cs, z * cs + r]))
            rr.log(f"world/trails/{name}",
                   rr.LineStrips3D([traj[: i + 1, 1:4] * cs], colors=[color], radii=0.08))


def main():
    ap = argparse.ArgumentParser(description="A: без ориентации")
    ap.add_argument("json", nargs="?")
    ap.add_argument("--save", metavar="FILE.rrd")
    a = ap.parse_args()
    data = json.load(open(a.json, encoding="utf-8")) if a.json else oc.demo_data_6d()
    rr.init("mapf_orient_a", spawn=not a.save)
    if a.save:
        rr.save(a.save)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_scene(data)


if __name__ == "__main__":
    main()