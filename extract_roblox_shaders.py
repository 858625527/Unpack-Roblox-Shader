"""Extract Roblox RBXS shader packs into individual GLSL source files.

Roblox's ``shaders_*.pack`` files are not ZIP files.  They contain a small
little-endian index followed by the shader source data.  This reader matches
the RBXS v11 layout used by the 0.735 client and validates every range before
writing output.
"""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path


MAGIC = 0x53584252  # b"RBXS" in little endian
HEADER = struct.Struct("<I6H")


def c_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip(" .")
    return name or "shader"


def read_pack(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("file is smaller than the RBXS header")
    magic, version, n_variants, n_flags, n_caps, n_names, n_files = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}; expected RBXS")
    if version != 11:
        raise ValueError(f"unsupported pack version {version}; this extractor expects v11")

    pos = HEADER.size
    # PackEntryVariant has a uint32 hash followed by a 64-byte variant name.
    end = pos + n_variants * 68
    if end > len(data):
        raise ValueError("truncated variant table")
    variants = [c_string(data[pos + i * 68 + 4 : pos + i * 68 + 68]) for i in range(n_variants)]
    pos = end

    # PackEntryShaderName is a 64-byte name followed by a uint32 pin mask.
    end = pos + n_names * 68
    if end > len(data):
        raise ValueError("truncated shader-name table")
    names = [c_string(data[pos + i * 68 : pos + i * 68 + 64]) for i in range(n_names)]
    pos = end

    end = pos + n_flags * 64
    if end > len(data):
        raise ValueError("truncated fast-flag table")
    flags = [c_string(data[pos + i * 64 : pos + (i + 1) * 64]) for i in range(n_flags)]
    pos = end

    end = pos + n_caps * 65
    if end > len(data):
        raise ValueError("truncated capability table")
    caps = [c_string(data[pos + i * 65 : pos + i * 65 + 64]) for i in range(n_caps)]
    pos = end

    files_end = pos + n_files * 64
    if files_end > len(data):
        raise ValueError("truncated file table")

    files = []
    for index in range(n_files):
        record = pos + index * 64
        # [0:16] is a content hash.  The loader uses the fields below:
        # source offset (+16), source size (+20), pin mask (+24),
        # variant index (+29), and shader-name index (+30).
        offset, size = struct.unpack_from("<II", data, record + 16)
        pin_mask = struct.unpack_from("<I", data, record + 24)[0]
        variant_index = data[record + 29]
        name_index = struct.unpack_from("<H", data, record + 30)[0]
        if name_index >= len(names):
            raise ValueError(f"file record {index} has invalid name index {name_index}")
        if offset + size > len(data):
            raise ValueError(
                f"file record {index} points outside pack: offset={offset}, size={size}"
            )
        files.append((index, names[name_index], pin_mask, variant_index, offset, size))

    return data, variants, flags, caps, files, files_end


def extract(pack: Path, output: Path) -> int:
    data, variants, flags, caps, files, _ = read_pack(pack)
    output.mkdir(parents=True, exist_ok=True)
    manifest = [
        "# Roblox RBXS shader pack",
        f"pack={pack}",
        f"version=11",
        f"variants={','.join(variants)}",
        f"files={len(files)}",
        "",
    ]
    for index, name, pin_mask, variant_index, offset, size in files:
        if variant_index >= len(variants):
            raise ValueError(f"file record {index} has invalid variant index {variant_index}")
        # This is the same eight hexadecimal digits appended by
        # RBX::Graphics::appendPinMask.  A zero mask is omitted.
        suffix = f"_{pin_mask:08x}" if pin_mask else ""
        filename = safe_name(name) + suffix + ".glsl"
        source = data[offset : offset + size]
        (output / filename).write_bytes(source)
        manifest.append(f"{filename}\t{name}\t{variants[variant_index]}\t{pin_mask:08x}\t{offset}\t{size}")
    (output / "manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path, help="path to shaders_glsl3.pack (or another RBXS v11 pack)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    count = extract(args.pack, args.output)
    print(f"extracted {count} shaders to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
