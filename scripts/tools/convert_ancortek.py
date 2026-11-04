"""
Convert Ancortek .bin captures to .npz files for the fmcw processing code.

Examples
--------
Single file, explicit output:
    python scripts/tools/convert_ancortek.py "data/radar_raw/corner reflector 1GHz/take_1.bin" -o data/radar_processed/1GHz_take_1.npz

Several files into radar_processed/ with automatic names:
    python scripts/tools/convert_ancortek.py "data/radar_raw/corner reflector 1GHz/"*.bin
"""

import argparse
import sys
from pathlib import Path

from fmcw.ancortek import read_ancortek_bin, save_capture


def default_output_name(input_path, out_dir):
    """Prefix the parent folder name so take_1.bin from different folders don't clash."""
    parent = input_path.parent.name.replace(" ", "_")
    return out_dir / f"{parent}_{input_path.stem}.npz"


def print_summary(params, cubes):
    print(f"  nTx={params['n_tx']}  nRx={params['n_rx']}")
    print(f"  sweep {params['f_start'] / 1e9:.3f} - {params['f_stop'] / 1e9:.3f} GHz "
          f"(B = {params['bandwidth'] / 1e6:.0f} MHz)")
    print(f"  T_sweep = {params['t_sweep'] * 1e6:.0f} us, "
          f"T_ramp = {params['t_ramp'] * 1e6:.1f} us, "
          f"N = {params['n_samp']}, fs = {params['fs'] / 1e3:.0f} kHz")
    print(f"  {params['n_sweeps']} sweeps per TX, {params['duration']:.2f} s")
    for name, cube in cubes.items():
        print(f"  {name}: shape {cube.shape} (N, M, L)")


def main():
    parser = argparse.ArgumentParser(description="Convert Ancortek .bin captures to .npz")
    parser.add_argument("inputs", type=Path, nargs="+", help=".bin file(s) to convert")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output .npz path (single input only)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/radar_processed"),
                        help="output folder when -o is not given (default: data/radar_processed)")
    parser.add_argument("--i-first", action="store_true",
                        help="treat sample pairs as [I, Q] instead of the vendor's [Q, I]")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing output files")
    args = parser.parse_args()

    if args.output is not None and len(args.inputs) > 1:
        parser.error("-o can only be used with a single input file; use --out-dir instead")

    failed = []

    for input_path in args.inputs:
        print(f"{input_path}")

        if input_path.suffix.lower() != ".bin":
            print("  skipped: not a .bin file")
            continue

        output = args.output or default_output_name(input_path, args.out_dir)
        if output.suffix != ".npz":
            output = output.with_suffix(".npz")

        if output.exists() and not args.overwrite:
            print(f"  skipped: {output} exists (use --overwrite)")
            continue

        try:
            params, cubes = read_ancortek_bin(input_path, q_first=not args.i_first)
        except ValueError as err:
            print(f"  failed: {err}")
            failed.append(input_path)
            continue

        print_summary(params, cubes)
        save_capture(output, params, cubes, source=input_path.name)
        print(f"  saved {output}")

    if failed:
        print(f"\n{len(failed)} file(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()