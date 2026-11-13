"""Self-contained ctypes wrapper for the Orbbec-patched OpenNI2 driver.

No third-party Python binding. Only needs `numpy` and the redist's
`libOpenNI2.so` (which auto-loads `OpenNI2/Drivers/liborbbec.so` relative to
its own directory, so callers should `chdir(REDIST)` before `init_context`).
"""
from __future__ import annotations
import ctypes as C
import os
import time
import numpy as np

ONI_API_VERSION = 2 * 1000 + 3
IR, COLOR, DEPTH = 1, 2, 3
PF_RGB888, PF_GRAY8, PF_GRAY16 = 200, 202, 203
PROP_HFOV, PROP_VFOV = 1, 2

class OniVideoMode(C.Structure):
    _fields_ = [("pixelFormat", C.c_int), ("resolutionX", C.c_int),
                ("resolutionY", C.c_int), ("fps", C.c_int)]

class OniDeviceInfo(C.Structure):
    _fields_ = [("uri", C.c_char * 256), ("vendor", C.c_char * 256),
                ("name", C.c_char * 256), ("usbVendorId", C.c_uint16),
                ("usbProductId", C.c_uint16)]

class OniFrame(C.Structure):
    _fields_ = [("dataSize", C.c_int), ("data", C.c_void_p),
                ("sensorType", C.c_int), ("timestamp", C.c_uint64),
                ("frameIndex", C.c_int), ("width", C.c_int), ("height", C.c_int),
                ("videoMode", OniVideoMode), ("croppingEnabled", C.c_int),
                ("cropOriginX", C.c_int), ("cropOriginY", C.c_int),
                ("stride", C.c_int)]


def get_redist() -> str:
    """Return the redist directory; honor $ORBBEC_REDIST, else look next to repo."""
    env = os.environ.get("ORBBEC_REDIST")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    guess = os.path.normpath(os.path.join(
        here, "..", ".driver", "ros2_astra_camera",
        "astra_camera", "openni2_redist", "x64"))
    return guess


class OniError(RuntimeError):
    pass


class Context:
    """OpenNI context + single-device handle. Use as a context manager."""

    def __init__(self, redist: str | None = None):
        self.redist = redist or get_redist()
        if not os.path.exists(os.path.join(self.redist, "libOpenNI2.so")):
            raise OniError(f"libOpenNI2.so missing at {self.redist} "
                           "(run setup.sh or set ORBBEC_REDIST)")
        os.chdir(self.redist)  # driver lookup is relative to cwd
        self.oni = C.CDLL(os.path.join(self.redist, "libOpenNI2.so"))
        self.oni.oniGetExtendedError.restype = C.c_char_p
        self.dev = C.c_void_p()
        self.info: OniDeviceInfo | None = None

    def _chk(self, status: int, what: str) -> None:
        if status != 0:
            err = self.oni.oniGetExtendedError()
            raise OniError(f"{what}: status={status} "
                           f"{err.decode(errors='replace') if err else ''}")

    def open(self, retries: int = 8, delay: float = 1.0) -> "Context":
        self._chk(self.oni.oniInitialize(ONI_API_VERSION), "oniInitialize")
        pdev = C.POINTER(OniDeviceInfo)(); n = C.c_int(0)
        self._chk(self.oni.oniGetDeviceList(C.byref(pdev), C.byref(n)),
                  "oniGetDeviceList")
        if n.value == 0:
            self.oni.oniShutdown()
            raise OniError("no OpenNI device found "
                           "(replug camera; check udev rules)")
        self.info = OniDeviceInfo()
        C.memmove(C.byref(self.info), pdev, C.sizeof(OniDeviceInfo))
        uri = pdev[0].uri
        self.oni.oniReleaseDeviceList(pdev)
        for i in range(retries):
            if self.oni.oniDeviceOpen(uri, C.byref(self.dev)) == 0:
                return self
            time.sleep(delay)
        self._chk(1, "oniDeviceOpen")
        return self

    def close(self) -> None:
        if self.dev:
            self.oni.oniDeviceClose(self.dev); self.dev = C.c_void_p()
        self.oni.oniShutdown()

    def __enter__(self): return self.open()
    def __exit__(self, *exc): self.close()


class Stream:
    """A single sensor stream (DEPTH or IR or COLOR)."""

    def __init__(self, ctx: Context, sensor: int):
        self.ctx, self.sensor = ctx, sensor
        self.handle = C.c_void_p()

    def start(self) -> "Stream":
        self.ctx._chk(self.ctx.oni.oniDeviceCreateStream(
            self.ctx.dev, self.sensor, C.byref(self.handle)), "createStream")
        self.ctx._chk(self.ctx.oni.oniStreamStart(self.handle), "streamStart")
        return self

    def stop(self) -> None:
        if self.handle:
            self.ctx.oni.oniStreamStop(self.handle)
            self.ctx.oni.oniStreamDestroy(self.handle)
            self.handle = C.c_void_p()

    def read(self) -> np.ndarray | None:
        pf = C.POINTER(OniFrame)()
        if self.ctx.oni.oniStreamReadFrame(self.handle, C.byref(pf)) != 0:
            return None
        fr = pf.contents
        raw = C.string_at(fr.data, fr.dataSize)
        pixfmt = fr.videoMode.pixelFormat
        if pixfmt == PF_RGB888:
            img = np.frombuffer(raw, np.uint8).reshape(fr.height, fr.width, 3)
        elif pixfmt == PF_GRAY8:
            img = np.frombuffer(raw, np.uint8).reshape(fr.height, fr.width)
        else:
            img = np.frombuffer(raw, np.uint16).reshape(fr.height, fr.width)
        out = img.copy()
        self.ctx.oni.oniFrameRelease(pf)
        return out

    def fov_rad(self) -> tuple[float, float]:
        h = C.c_float(0); v = C.c_float(0); sz = C.c_int(C.sizeof(h))
        self.ctx.oni.oniStreamGetProperty(self.handle, PROP_HFOV, C.byref(h), C.byref(sz))
        self.ctx.oni.oniStreamGetProperty(self.handle, PROP_VFOV, C.byref(v), C.byref(sz))
        return h.value, v.value

    def __enter__(self): return self.start()
    def __exit__(self, *exc): self.stop()
