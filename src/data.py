from dataclasses import dataclass
import h5py
import numpy as np

@dataclass
class Data:
    specs_OFF: np.ndarray
    specs_DIFF: np.ndarray
    specs_zf_OFF: np.ndarray
    specs_zf_DIFF: np.ndarray

    ppm_array: np.ndarray
    ppm_array_zf: np.ndarray

    n: int
    spectral_width: float
    Bo: float
    dwell_time: float
    te: float
    tr: float

    vendor: str

    @classmethod
    def load_h5(cls, filename: str):
        with h5py.File(filename, "r") as f:
            return cls(
                specs_OFF=f["specs_OFF"][()],
                specs_DIFF=f["specs_DIFF"][()],
                specs_zf_OFF=f["specs_zf_OFF"][()],
                specs_zf_DIFF=f["specs_zf_DIFF"][()],

                ppm_array=f["ppm_array"][()],
                ppm_array_zf=f["ppm_array_zf"][()],

                n=int(f["n"][()]),
                spectral_width=float(f["spectral_width"][()]),
                Bo=float(f["Bo"][()]),
                dwell_time=float(f["dwell_time"][()]),
                te=float(f["te"][()]),
                tr=float(f["tr"][()]),

                vendor=f["vendor"][()].decode("utf-8"),
            )