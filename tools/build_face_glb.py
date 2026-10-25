"""GA3 Q6 — build a textured 3D head from a single photograph.

The head is a sculpted ellipsoid: a generic skull shape carries brow, eye
socket, nose, cheek, lip and chin features, and a heavily blurred relief term
taken from the photograph's own shading pulls those features towards this
particular face. The photograph is reprojected into a spherical texture so the
front reads as the person and the sides and back stay plausible hair.

Usage: python tools/build_face_glb.py <photo.jpg> <out.glb>
"""
from __future__ import annotations

import json
import struct
import sys

import numpy as np
from PIL import Image, ImageFilter

# Face region of the photograph, as fractions of width/height: the ellipse that
# the head's silhouette maps onto.
FACE_CX, FACE_CY = 0.470, 0.360
FACE_HALF_W, FACE_HALF_H = 0.234, 0.300

NU, NV = 96, 64            # azimuth and polar grid
TEXTURE = 1024
RADII = (0.78, 1.00, 0.88)  # x (width), y (height), z (depth)


def gaussian(dx, dy, sx, sy):
    return np.exp(-((dx / sx) ** 2 + (dy / sy) ** 2))


def sculpt(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Depth added to the front of the head, in model units."""
    front = np.clip(z, 0, None)                      # only the facing side
    weight = front ** 1.2
    depth = np.zeros_like(x)

    # brow ridge and forehead slope
    depth += 0.045 * gaussian(x, y - 0.30, 0.42, 0.09) * weight
    depth -= 0.030 * gaussian(x, y - 0.62, 0.45, 0.22) * weight
    # eye sockets
    for side in (-1, 1):
        depth -= 0.055 * gaussian(x - side * 0.26, y - 0.19, 0.15, 0.09) * weight
    # nose: bridge, body and tip
    nose = gaussian(x, y - 0.05, 0.075, 0.26)
    depth += 0.115 * nose * weight
    depth += 0.055 * gaussian(x, y + 0.06, 0.085, 0.07) * weight
    for side in (-1, 1):                              # nostril wings
        depth += 0.020 * gaussian(x - side * 0.11, y + 0.07, 0.05, 0.05) * weight
    # cheekbones
    for side in (-1, 1):
        depth += 0.042 * gaussian(x - side * 0.40, y - 0.02, 0.20, 0.15) * weight
    # mouth: lips with a crease between them
    depth += 0.040 * gaussian(x, y + 0.30, 0.19, 0.07) * weight
    depth -= 0.022 * gaussian(x, y + 0.31, 0.17, 0.018) * weight
    # chin and the shadow above it
    depth += 0.050 * gaussian(x, y + 0.56, 0.17, 0.11) * weight
    depth -= 0.020 * gaussian(x, y + 0.44, 0.15, 0.05) * weight
    # temples pulled in
    for side in (-1, 1):
        depth -= 0.030 * gaussian(x - side * 0.62, y - 0.34, 0.14, 0.14) * weight
    return depth


def relief_map(photo: Image.Image) -> np.ndarray:
    """Blurred luminance of the face region, zero-mean, as a relief term."""
    width, height = photo.size
    box = (
        int((FACE_CX - FACE_HALF_W) * width),
        int((FACE_CY - FACE_HALF_H) * height),
        int((FACE_CX + FACE_HALF_W) * width),
        int((FACE_CY + FACE_HALF_H) * height),
    )
    face = photo.convert("L").crop(box).resize((128, 128), Image.LANCZOS)
    blurred = face.filter(ImageFilter.GaussianBlur(radius=6))
    values = np.asarray(blurred, dtype=np.float32) / 255.0
    values -= values.mean()
    spread = np.abs(values).max() or 1.0
    return values / spread


def sample_relief(relief: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample the relief map at normalised face coordinates."""
    size = relief.shape[0]
    cols = np.clip(((x + 1) / 2) * (size - 1), 0, size - 1).astype(np.int32)
    rows = np.clip(((1 - y) / 2) * (size - 1), 0, size - 1).astype(np.int32)
    return relief[rows, cols]


def build_mesh(photo: Image.Image):
    relief = relief_map(photo)
    u = np.linspace(0.0, 1.0, NU + 1)
    v = np.linspace(0.0, 1.0, NV + 1)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    theta = (grid_u - 0.5) * 2 * np.pi          # 0 at the front
    phi = grid_v * np.pi                        # 0 at the top

    dir_x = np.sin(theta) * np.sin(phi)
    dir_y = np.cos(phi)
    dir_z = np.cos(theta) * np.sin(phi)

    rx, ry, rz = RADII
    # jaw taper: narrower below the cheekbones, wider across the cranium
    taper = 1.0 + 0.16 * np.clip(dir_y, 0, None) - 0.22 * np.clip(-dir_y - 0.25, 0, None)
    # the hair sits proudest above the ears and at the back of the head
    hair = 1.0 + 0.17 * np.clip(dir_y + 0.15, 0, None) * np.clip(1.0 - dir_z, 0, 2) / 2
    x = dir_x * rx * taper * hair
    y = dir_y * ry * (1.0 + 0.05 * np.clip(dir_y, 0, None))
    z = dir_z * rz * hair
    # occipital bulge: a real skull is longer behind the ears than a sphere
    z = z - 0.10 * np.clip(-dir_z, 0, None) ** 1.5 * (1.0 - np.clip(dir_y, 0, None) * 0.4)

    face_x = np.clip(x / rx, -1, 1)
    face_y = np.clip(y / ry, -1, 1)
    front = np.clip(dir_z, 0, None)
    depth = sculpt(face_x, face_y, dir_z)
    depth += 0.055 * sample_relief(relief, face_x, face_y) * (front ** 2)
    z = z + depth

    # jawline: pull the chin forward so the profile reads as a face
    z = z + 0.06 * np.clip(dir_z, 0, None) * gaussian(face_x, face_y + 0.68, 0.30, 0.16)

    positions = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    uvs = np.stack([grid_u, grid_v], axis=-1).reshape(-1, 2).astype(np.float32)

    indices = []
    for i in range(NU):
        for j in range(NV):
            a = i * (NV + 1) + j
            b = (i + 1) * (NV + 1) + j
            c = (i + 1) * (NV + 1) + j + 1
            d = i * (NV + 1) + j + 1
            # counter-clockwise when seen from outside, which is what glTF
            # treats as the front face
            if j != 0:
                indices.extend((a, c, b))
            if j != NV - 1:
                indices.extend((a, d, c))
    indices = np.array(indices, dtype=np.uint32).reshape(-1, 3)

    normals = np.zeros_like(positions)
    tri = positions[indices]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for column in range(3):
        np.add.at(normals, indices[:, column], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = (normals / lengths).astype(np.float32)
    return positions, normals, uvs, indices.reshape(-1).astype(np.uint32)


def build_texture(photo: Image.Image) -> Image.Image:
    """Reproject the photograph into the head's spherical UV space."""
    pixels = np.asarray(photo.convert("RGB"), dtype=np.float32)
    height, width, _ = pixels.shape

    u = (np.arange(TEXTURE) + 0.5) / TEXTURE
    v = (np.arange(TEXTURE) + 0.5) / TEXTURE
    grid_u, grid_v = np.meshgrid(u, v, indexing="xy")
    theta = (grid_u - 0.5) * 2 * np.pi
    phi = grid_v * np.pi
    dir_x = np.sin(theta) * np.sin(phi)
    dir_y = np.cos(phi)
    dir_z = np.cos(theta) * np.sin(phi)

    # Orthographic projection of the sphere onto the photograph's face ellipse;
    # the far side mirrors back onto the photo and is darkened into hair.
    px = (FACE_CX + dir_x * FACE_HALF_W) * (width - 1)
    py = (FACE_CY - dir_y * FACE_HALF_H) * (height - 1)
    cols = np.clip(px, 0, width - 1).astype(np.int32)
    rows = np.clip(py, 0, height - 1).astype(np.int32)
    texture = pixels[rows, cols]

    # Past roughly 55 degrees of azimuth the orthographic projection compresses
    # the photograph into rings, so blend it out into a hair shell sampled from
    # the photograph's own hair rather than folding the image back on itself.
    hair_box = pixels[
        int(0.16 * height) : int(0.34 * height), int(0.20 * width) : int(0.76 * width)
    ]
    hair_colour = hair_box.reshape(-1, 3).mean(axis=0) * 0.55

    away = np.degrees(np.arccos(np.clip(dir_z, -1, 1)))[..., None]
    blend = np.clip((away - 66.0) / 26.0, 0.0, 1.0) ** 1.3
    # a little vertical banding keeps the hair shell from looking like plastic
    banding = 1.0 + 0.025 * np.sin(grid_v[..., None] * np.pi * 18) * (1 - np.cos(theta)[..., None])
    shell = hair_colour * banding
    # the crown keeps real hair texture, which is what a viewer sees from above
    crown = np.clip((0.78 - grid_v) / 0.35, 0.0, 1.0)[..., None]
    blend = blend * (1.0 - 0.55 * crown * np.clip(1 - away / 110.0, 0, 1))
    texture = texture * (1 - blend) + shell * blend
    return Image.fromarray(np.clip(texture, 0, 255).astype(np.uint8))


def pad(data: bytes) -> bytes:
    return data + b"\x00" * (-len(data) % 4)


def write_glb(path, positions, normals, uvs, indices, texture: Image.Image) -> None:
    import io

    jpeg = io.BytesIO()
    texture.save(jpeg, format="JPEG", quality=84, optimize=True, progressive=False)
    jpeg_bytes = jpeg.getvalue()

    index_data = indices.astype(np.uint32).tobytes()
    chunks = [
        ("indices", index_data),
        ("positions", positions.tobytes()),
        ("normals", normals.tobytes()),
        ("uvs", uvs.tobytes()),
        ("image", jpeg_bytes),
    ]
    binary = b""
    views = []
    for name, data in chunks:
        offset = len(binary)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        binary += pad(data)
    views[0]["target"] = 34963   # ELEMENT_ARRAY_BUFFER
    for index in (1, 2, 3):
        views[index]["target"] = 34962  # ARRAY_BUFFER

    gltf = {
        "asset": {"version": "2.0", "generator": "tds-2026-09-ga3 face builder"},
        "scene": 0,
        "scenes": [{"nodes": [0], "name": "Scene"}],
        "nodes": [{"mesh": 0, "name": "Head"}],
        "meshes": [
            {
                "name": "Head",
                "primitives": [
                    {
                        "attributes": {"POSITION": 1, "NORMAL": 2, "TEXCOORD_0": 3},
                        "indices": 0,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "Skin",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0, "texCoord": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.85,
                },
                "doubleSided": False,
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 33071}],
        "images": [{"bufferView": 4, "mimeType": "image/jpeg", "name": "face"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": int(indices.size), "type": "SCALAR",
             "max": [int(indices.max())], "min": [int(indices.min())]},
            {"bufferView": 1, "componentType": 5126, "count": int(len(positions)), "type": "VEC3",
             "max": positions.max(axis=0).tolist(), "min": positions.min(axis=0).tolist()},
            {"bufferView": 2, "componentType": 5126, "count": int(len(normals)), "type": "VEC3"},
            {"bufferView": 3, "componentType": 5126, "count": int(len(uvs)), "type": "VEC2"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }

    json_bytes = pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    json_bytes = json_bytes.replace(b"\x00", b"\x20") if json_bytes[-1:] == b"\x00" else json_bytes
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(json_chunk) + len(bin_chunk))
    with open(path, "wb") as handle:
        handle.write(header + json_chunk + bin_chunk)


def main() -> int:
    photo = Image.open(sys.argv[1])
    positions, normals, uvs, indices = build_mesh(photo)
    texture = build_texture(photo)
    write_glb(sys.argv[2], positions, normals, uvs, indices, texture)
    import os

    print(f"{sys.argv[2]}: {os.path.getsize(sys.argv[2])} bytes, "
          f"{len(positions)} vertices, {indices.size // 3} triangles")
    np.save("/tmp/ga3-work/mesh.npy", np.array([positions, normals], dtype=object), allow_pickle=True)
    np.save("/tmp/ga3-work/mesh_idx.npy", indices)
    np.save("/tmp/ga3-work/mesh_uv.npy", uvs)
    texture.save("/tmp/ga3-work/texture.jpg", quality=84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
