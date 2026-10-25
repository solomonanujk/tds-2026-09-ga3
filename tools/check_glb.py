"""Validate a GLB and render it from several angles with a small software
rasteriser, so the geometry and texture can be inspected without a viewer."""
from __future__ import annotations

import io
import json
import struct
import sys

import numpy as np
from PIL import Image

COMPONENT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def parse(path):
    data = open(path, "rb").read()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "not a GLB"
    assert version == 2, f"glTF version {version}"
    assert length == len(data), f"header length {length} != file {len(data)}"
    offset, chunks = 12, []
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        chunks.append((chunk_type, data[offset + 8 : offset + 8 + chunk_length]))
        assert chunk_length % 4 == 0, "chunk length not 4-byte aligned"
        offset += 8 + chunk_length
    assert chunks[0][0] == 0x4E4F534A, "first chunk must be JSON"
    assert chunks[1][0] == 0x004E4942, "second chunk must be BIN"
    return json.loads(chunks[0][1].decode("utf-8")), chunks[1][1]


def accessor(gltf, binary, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    fmt = COMPONENT[acc["componentType"]]
    size = COUNT[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    array = np.frombuffer(binary, dtype=np.dtype(f"<{fmt}"), count=acc["count"] * size, offset=start)
    return array.reshape(acc["count"], size) if size > 1 else array


def render(positions, normals, uvs, indices, texture, angle_deg, size=360):
    angle = np.radians(angle_deg)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    pts = positions @ rotation.T
    nrm = normals @ rotation.T
    scale = size * 0.40
    sx = size / 2 + pts[:, 0] * scale
    sy = size / 2 - pts[:, 1] * scale
    depth = pts[:, 2]

    image = np.full((size, size, 3), 24, dtype=np.float32)
    zbuf = np.full((size, size), -1e9, dtype=np.float32)
    tex = np.asarray(texture, dtype=np.float32)
    th, tw, _ = tex.shape
    light = np.array([0.35, 0.45, 0.82])
    light = light / np.linalg.norm(light)

    for tri in indices.reshape(-1, 3):
        xs, ys, zs = sx[tri], sy[tri], depth[tri]
        area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if area >= -1e-6:          # back-face cull (screen y is flipped)
            continue
        min_x, max_x = int(max(xs.min(), 0)), int(min(xs.max() + 1, size))
        min_y, max_y = int(max(ys.min(), 0)), int(min(ys.max() + 1, size))
        if min_x >= max_x or min_y >= max_y:
            continue
        gy, gx = np.mgrid[min_y:max_y, min_x:max_x]
        px, py = gx + 0.5, gy + 0.5
        w0 = ((xs[1] - xs[0]) * (py - ys[0]) - (px - xs[0]) * (ys[1] - ys[0])) / area
        w1 = ((px - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (py - ys[0])) / area
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w2 * zs[0] + w1 * zs[1] + w0 * zs[2]
        rows, cols = gy[inside], gx[inside]
        closer = z[inside] > zbuf[rows, cols]
        rows, cols = rows[closer], cols[closer]
        if rows.size == 0:
            continue
        bw0, bw1, bw2 = w0[inside][closer], w1[inside][closer], w2[inside][closer]
        zbuf[rows, cols] = z[inside][closer]
        u = bw2 * uvs[tri[0], 0] + bw1 * uvs[tri[1], 0] + bw0 * uvs[tri[2], 0]
        v = bw2 * uvs[tri[0], 1] + bw1 * uvs[tri[1], 1] + bw0 * uvs[tri[2], 1]
        colour = tex[np.clip((v * (th - 1)).astype(int), 0, th - 1),
                     np.clip((u * (tw - 1)).astype(int), 0, tw - 1)]
        normal = (bw2[:, None] * nrm[tri[0]] + bw1[:, None] * nrm[tri[1]] + bw0[:, None] * nrm[tri[2]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        shade = 0.78 + 0.32 * np.clip(normal @ light, 0, 1)
        image[rows, cols] = np.clip(colour * shade[:, None], 0, 255)
    return Image.fromarray(image.astype(np.uint8))


def main() -> int:
    gltf, binary = parse(sys.argv[1])
    mesh = gltf["meshes"][0]["primitives"][0]
    positions = accessor(gltf, binary, mesh["attributes"]["POSITION"]).astype(np.float32)
    normals = accessor(gltf, binary, mesh["attributes"]["NORMAL"]).astype(np.float32)
    uvs = accessor(gltf, binary, mesh["attributes"]["TEXCOORD_0"]).astype(np.float32)
    indices = accessor(gltf, binary, mesh["indices"]).astype(np.int64)
    image_def = gltf["images"][0]
    view = gltf["bufferViews"][image_def["bufferView"]]
    texture = Image.open(io.BytesIO(binary[view["byteOffset"]: view["byteOffset"] + view["byteLength"]])).convert("RGB")

    assert gltf["asset"]["version"] == "2.0"
    assert indices.max() < len(positions), "index out of range"
    acc = gltf["accessors"][mesh["attributes"]["POSITION"]]
    assert np.allclose(acc["min"], positions.min(axis=0), atol=1e-5), "POSITION min wrong"
    assert np.allclose(acc["max"], positions.max(axis=0), atol=1e-5), "POSITION max wrong"
    print(f"valid glTF 2.0 GLB: {len(positions)} verts, {indices.size // 3} tris, "
          f"texture {texture.size}, material {gltf['materials'][0]['name']}")

    strip = Image.new("RGB", (360 * 5, 360))
    for column, angle in enumerate((-60, -30, 0, 30, 60)):
        strip.paste(render(positions, normals, uvs, indices, texture, angle), (column * 360, 0))
    strip.save(sys.argv[2])
    print("wrote", sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
