from ..world import (
    World,
    WorldEdge,
)

type_b_circuit = World()
center_xy = (6.26263, 9.47251)
rotate_deg = 180

# centerは追従用なので厳密に連続でなくてよいのでauto_connectは使用しない（使用すると外乱になる）
lane_center = WorldEdge(
    "lane_center", "black", 1, "dash", center_xy=center_xy, rotate_deg=rotate_deg
)
lane_center.add_line((6.26263, 9.47251), (3.08263, 9.47251))
lane_center.add_arc((3.08263, 8.36251), 1.11, 90, 255)
lane_center.add_arc((2.89177, 7.67853), 0.4, 256.051, 34.9202)
lane_center.add_arc((3.55343, 8.01312), 0.35, 197.564, 93.407, inverse=True)
lane_center.add_line((3.53263, 8.36251), (6.31012, 8.70969))
lane_center.add_arc((6.26263, 9.08962), 0.382885, 277.125, 90)

# outerは内外判定用なので厳密に連続である必要がある。その為、auto_connect=Trueを使用している
lane_outer = WorldEdge(
    "lane_outer", "blue", 2, "solid", center_xy=center_xy, rotate_deg=rotate_deg
)
lane_outer.add_line((6.26263, 9.67251), (3.08263, 9.67251))
lane_outer.add_arc((3.08263, 8.36251), 1.31, 90, 255.075, auto_connect=True)
lane_outer.add_arc((2.89177, 7.67853), 0.6, 255.865, 31.1137, auto_connect=True)
lane_outer.add_arc(
    (3.55343, 8.01312), 0.15, 189.42, 102.806, inverse=True, auto_connect=True
)
lane_outer.add_line((3.52019, 8.15939), (3.94263, 8.2122), auto_connect=True)
lane_outer.add_line((3.94263, 8.2122), (4.04263, 8.32548))
lane_outer.add_line((4.04263, 8.32548), (4.84263, 8.42548))
lane_outer.add_line((4.84263, 8.42548), (4.94263, 8.3372))
lane_outer.add_line((4.94263, 8.3372), (6.33493, 8.51124))
lane_outer.add_arc((6.26263, 9.08962), 0.582885, 277.125, 90, auto_connect=True)
lane_outer.auto_close()

# innerは内外判定用なので厳密に連続である必要がある。その為、auto_connect=Trueを使用している
lane_inner = WorldEdge(
    "lane_inner", "green", 2, "solid", center_xy=center_xy, rotate_deg=rotate_deg
)
lane_inner.add_line((6.26263, 9.27251), (3.08263, 9.27251))
lane_inner.add_arc((3.08263, 8.36251), 0.91, 90, 254.87, auto_connect=True)
lane_inner.add_arc((2.89177, 7.67853), 0.2, 256.511, 41.2345, auto_connect=True)
lane_inner.add_arc(
    (3.55343, 8.01312), 0.55, 201.633, 93.407, inverse=True, auto_connect=True
)
lane_inner.add_line((3.50783, 8.56096), (3.94263, 8.61531), auto_connect=True)
lane_inner.add_line((3.94263, 8.61531), (4.04263, 8.52703))
lane_inner.add_line((4.04263, 8.52703), (4.84263, 8.62703))
lane_inner.add_line((4.84263, 8.62703), (4.94263, 8.74031))
lane_inner.add_line((4.94263, 8.74031), (6.28532, 8.90815))
lane_inner.add_arc((6.26263, 9.08962), 0.182885, 277.125, 90, auto_connect=True)
lane_inner.auto_close()

rim = WorldEdge("rim", "black", 1, center_xy=center_xy, rotate_deg=rotate_deg)
rim.add_lines(
    [(7.17263, 9.72751), (1.71263, 9.72751), (1.71263, 6.99751), (7.17263, 6.99751)],
    loop=True,
)
start = WorldEdge(
    "start", "white", 1, "solid", True, center_xy=center_xy, rotate_deg=rotate_deg
)
start_half_width = 0.03
start.add_lines(
    [
        (3.53263 + start_half_width, 9.67251),
        (3.53263 - start_half_width, 9.67251),
        (3.53263 - start_half_width, 9.27251),
        (3.53263 + start_half_width, 9.27251),
    ],
    loop=True,
)

type_b_circuit.add_edges([lane_center, lane_inner, lane_outer, rim, start])
type_b_circuit.set_auto_edge("lane_center", "lane_inner", "lane_outer")
