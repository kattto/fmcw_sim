"""
Reader for Ancortek SDR-KIT 2400T2R4 .bin captures (standalone GUI recordings).

File layout:

- Header: six little-endian uint16 words
    1: nRx
    2: nTx
    3: start frequency (MHz)
    4: stop frequency (MHz)
    5: sweep time (us)
    6: samples per sweep
- Data: uint16 ADC counts, interleaved per time sample as
    [Q1, I1, Q2, I2, Q3, I3, Q4, I4]     (4 RX shown)
  with sweeps stored back to back.
- The first value of every sweep carries a marker:
    +49152 (0xC000) on TX1 sweeps
    +32768 (0x8000) on TX2 sweeps
  In 2-TX mode, TX1 and TX2 sweeps alternate.

The vendor comments say [I, Q] but all of the vendor's working extraction code
treats the marker sample as Q. q_first=True follows the code.

The data is raw and unwindowed.
"""

import warnings
from pathlib import Path

import numpy as np

TX1_MARKER = 49152
TX2_MARKER = 32768
HEADER_WORDS = 6

# Down-chirp duration per sweep time, from PLL_registry_calculation in the
# vendor GUI. Keys are sweep times in microseconds.
_DOWNCHIRP_TIME = {
    125: 5e-6,
    250: 5e-6,
    500: 70e-6,
    1000: 90e-6,
    2000: 110e-6,
    4000: 90e-6,
    8000: 90e-6,
}


def ramp_time(t_sweep):
    """Duration of the up-chirp within one sweep period, in seconds.

    Follows the vendor PLL setup: T_ramp = T_sweep - 1.5 * T_downchirp.
    Returns NaN for sweep times not in the vendor table.
    Verify against the data: the end of the ramp shows up as a discontinuity
    in the time-domain sweep.
    """
    key = int(round(t_sweep * 1e6))
    if key not in _DOWNCHIRP_TIME:
        return float("nan")
    return t_sweep - 1.5 * _DOWNCHIRP_TIME[key]


def read_header(raw):
    """Parse the six-word header into a parameter dictionary."""
    if raw.size < HEADER_WORDS:
        raise ValueError("file is too short to contain a header")

    n_rx, n_tx, f_start_mhz, f_stop_mhz, t_sweep_us, n_samp = (int(v) for v in raw[:HEADER_WORDS])

    if n_rx not in (1, 2, 4):
        raise ValueError(f"header nRx={n_rx} is not 1, 2 or 4")
    if n_tx not in (1, 2):
        raise ValueError(f"header nTx={n_tx} is not 1 or 2")
    if n_samp <= 0:
        raise ValueError(f"header samples per sweep={n_samp} is not positive")
    if f_stop_mhz < f_start_mhz:
        raise ValueError(f"header stop frequency {f_stop_mhz} MHz is below start {f_start_mhz} MHz")

    t_sweep = t_sweep_us * 1e-6

    return {
        "n_rx": n_rx,
        "n_tx": n_tx,
        "f_start": f_start_mhz * 1e6,
        "f_stop": f_stop_mhz * 1e6,
        "bandwidth": (f_stop_mhz - f_start_mhz) * 1e6,
        "t_sweep": t_sweep,
        "t_ramp": ramp_time(t_sweep),
        "n_samp": n_samp,
        "fs": n_samp / t_sweep,
    }


def find_markers(adc):
    """Indices of TX1 and TX2 sweep markers in the ADC stream."""
    idx_tx1 = np.flatnonzero(adc >= TX1_MARKER)
    idx_tx2 = np.flatnonzero((adc >= TX2_MARKER) & (adc < TX1_MARKER))
    return idx_tx1, idx_tx2


def read_ancortek_bin(path, q_first=True):
    """Read an Ancortek .bin capture.

    Returns
    -------
    params : dict
        Header values plus derived quantities and bookkeeping.
    cubes : dict
        {"tx1": cube} or {"tx1": cube, "tx2": cube}. Each cube is complex64
        with shape (N, M, L): fast-time samples, sweeps, RX channels.
    """
    raw = np.fromfile(path, dtype="<u2")
    params = read_header(raw)

    n_rx = params["n_rx"]
    n_tx = params["n_tx"]
    n_samp = params["n_samp"]

    adc = raw[HEADER_WORDS:].astype(np.int64)

    values_per_sweep = n_samp * n_rx * 2       # samples x RX x (Q, I)
    frame = values_per_sweep * n_tx            # one TX1 sweep (+ one TX2 sweep)

    # --- markers and TX mode cross-check
    idx_tx1, idx_tx2 = find_markers(adc)
    if idx_tx1.size == 0:
        raise ValueError("no TX1 sweep markers found")

    markers_say_two_tx = idx_tx2.size > 0
    if markers_say_two_tx != (n_tx == 2):
        raise ValueError(
            f"header says nTx={n_tx} but the marker pattern says "
            f"nTx={2 if markers_say_two_tx else 1}"
        )

    start = int(idx_tx1[0])

    # --- keep the contiguous run of frames before any dropped data
    n_frames = (adc.size - start) // frame

    gaps = np.flatnonzero(np.diff(idx_tx1) != frame)
    if gaps.size > 0:
        n_frames = min(n_frames, int(gaps[0]) + 1)
        warnings.warn(
            f"irregular TX1 marker spacing (data loss); keeping the first "
            f"{n_frames} contiguous frames only"
        )

    if n_tx == 2:
        expected_tx2 = start + values_per_sweep + np.arange(n_frames) * frame
        present = np.isin(expected_tx2, idx_tx2)
        if not present.all():
            n_frames = int(np.argmin(present))
            warnings.warn(
                f"missing TX2 marker; keeping the first {n_frames} contiguous frames only"
            )

    if n_frames == 0:
        raise ValueError("no complete sweeps found")

    # --- remove markers
    adc[idx_tx1] -= TX1_MARKER
    adc[idx_tx2] -= TX2_MARKER

    # --- reshape: (frame, TX, sample, RX, pair)
    block = adc[start:start + n_frames * frame]
    block = block.reshape(n_frames, n_tx, n_samp, n_rx, 2)

    if q_first:
        i_part = block[..., 1]
        q_part = block[..., 0]
    else:
        i_part = block[..., 0]
        q_part = block[..., 1]

    iq = (i_part + 1j * q_part).astype(np.complex64)

    cubes = {}
    for t in range(n_tx):
        # (sweep, sample, RX) -> (sample, sweep, RX) = (N, M, L)
        cubes[f"tx{t + 1}"] = np.ascontiguousarray(iq[:, t].transpose(1, 0, 2))

    params["n_sweeps"] = n_frames                       # per TX
    params["duration"] = n_frames * n_tx * params["t_sweep"]
    params["leading_values_skipped"] = start
    params["q_first"] = q_first

    return params, cubes


def save_capture(path, params, cubes, source):
    """Save a converted capture as .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {"source": np.array(source)}
    for key, value in params.items():
        arrays[f"param_{key}"] = np.array(value)
    for key, cube in cubes.items():
        arrays[f"cube_{key}"] = cube

    np.savez(path, **arrays)


def load_capture(path):
    """Load a capture saved by save_capture.

    Returns (params, cubes, source).
    """
    with np.load(path) as data:
        source = str(data["source"])
        params = {
            key[len("param_"):]: data[key].item()
            for key in data.files if key.startswith("param_")
        }
        cubes = {
            key[len("cube_"):]: data[key]
            for key in data.files if key.startswith("cube_")
        }
    return params, cubes, source