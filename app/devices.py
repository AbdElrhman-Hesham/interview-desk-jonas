"""Audio input device enumeration and sane default selection.

Deliberately small. This is NOT an audio control panel -- it answers one
question: "which microphone should Lena listen to, and is it a bad one?"

⚑ WHY THIS EXISTS: the environment work found that Bluetooth headsets enumerate
through `bthhfenum.sys` at 8000 Hz narrowband the moment the microphone is
opened (HFP profile). Whisper is trained on 16 kHz audio; feeding it 8 kHz
telephone-band speech destroys recognition of exactly the sounds a German
learner needs distinguished. Silently picking such a device would make the
product look broken for reasons the user cannot see.
"""
from __future__ import annotations

# Anything at or below this default sample rate is telephone-band and will hurt
# recognition quality. 16000 is Whisper's native rate; a device *defaulting* to
# 16k or less is a strong HFP/narrowband signal on Windows.
NARROWBAND_HZ = 16000

_HFP_HINTS = ("hands-free", "handsfree", "headset", "hfp", "bluetooth", "bth")


def list_inputs() -> list[dict]:
    """All usable input devices, annotated. Never raises."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return []

    out = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) < 1:
            continue
        name = d.get("name", f"device {i}")
        sr = int(d.get("default_samplerate") or 0)
        low = name.lower()
        hfp = any(h in low for h in _HFP_HINTS)
        narrow = sr and sr <= NARROWBAND_HZ
        try:
            api = hostapis[d["hostapi"]]["name"]
        except Exception:
            api = ""
        out.append({
            "index": i,
            "name": name,
            "sample_rate": sr or None,
            "channels": d.get("max_input_channels"),
            "hostapi": api,
            "narrowband": bool(narrow),
            "likely_bluetooth_hfp": bool(hfp and narrow),
            # what the UI shows when a device is a bad idea
            "warning": ("Telefonqualität (8 kHz) — Spracherkennung wird schlecht"
                        if narrow else None),
        })
    return out


def default_input(devices: list[dict] | None = None) -> int | None:
    """Pick a sensible microphone.

    Preference order:
      1. the system default input, if it is not narrowband
      2. the widest-band non-narrowband device
      3. anything at all (with the warning already attached by list_inputs)

    Returns a device index, or None to let PortAudio choose.
    """
    devs = list_inputs() if devices is None else devices
    if not devs:
        return None

    good = [d for d in devs if not d["narrowband"]]

    try:
        import sounddevice as sd
        sys_default = sd.default.device[0]
    except Exception:
        sys_default = None

    if sys_default is not None and sys_default >= 0:
        for d in good:
            if d["index"] == sys_default:
                return d["index"]

    if good:
        # widest band wins; ties broken by lowest index (usually the built-in)
        return sorted(good, key=lambda d: (-(d["sample_rate"] or 0), d["index"]))[0]["index"]
    return devs[0]["index"]


def describe(index: int | None) -> str:
    if index is None:
        return "Systemstandard"
    for d in list_inputs():
        if d["index"] == index:
            return d["name"]
    return f"Gerät {index}"
