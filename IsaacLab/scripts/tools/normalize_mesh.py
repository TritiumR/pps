"""Normalize an OBJ mesh so that the bounding-box center becomes the origin.

Only vertex position lines ('v x y z') are modified; everything else
(mtllib, usemtl, texture coords, normals, faces, etc.) is preserved as-is.
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Center an OBJ mesh at the bounding-box midpoint.")
    parser.add_argument("input", help="Path to the input .obj file")
    parser.add_argument("-o", "--output", default=None, help="Path to the output .obj file (default: overwrite input)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = f.readlines()

    vertices = []
    vertex_line_indices = []
    for i, line in enumerate(lines):
        parts = line.split()
        if parts and parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            vertex_line_indices.append(i)

    if not vertices:
        raise RuntimeError(f"No vertices found in {args.input}")

    verts = np.array(vertices)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts -= center

    for idx, vi in enumerate(vertex_line_indices):
        parts = lines[vi].split()
        extra = " ".join(parts[4:])  # preserve optional w or color data
        new_line = f"v {verts[idx][0]:.8g} {verts[idx][1]:.8g} {verts[idx][2]:.8g}"
        if extra:
            new_line += " " + extra
        lines[vi] = new_line + "\n"

    out_path = args.output or args.input
    with open(out_path, "w") as f:
        f.writelines(lines)

    print(f"Centered mesh written to {out_path}  (shifted by {center})")


if __name__ == "__main__":
    main()
