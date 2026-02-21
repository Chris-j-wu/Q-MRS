from dataclasses import dataclass
import h5py
import numpy as np

# @dataclass
# class Car:
#     """A simple class to represent a car."""
#     brand: str
#     model: str
#     color: str

#     def display_info(self):
#         """Print the car's information."""
#         print(f"Brand: {self.brand}, Model: {self.model}, Color: {self.color}")

@dataclass
class BasisSet:
    fids_OFF: np.ndarray
    fids_DIFF: np.ndarray
    fids_zf_OFF: np.ndarray
    fids_zf_DIFF: np.ndarray

    ppm_array: np.ndarray
    ppm_array_zf: np.ndarray

    n: int
    spectral_width: float
    Bo: float
    dwell_time: float
    te: float
    vendor: str

    metab_names_OFF: list
    MM_names_OFF: list
    metab_names_DIFF: list
    MM_names_DIFF: list

    @classmethod
    def load_h5(cls, filename: str):
        with h5py.File(filename, "r") as f:
            return cls(
                fids_OFF=f["fids_OFF"][()],
                fids_DIFF=f["fids_DIFF"][()],
                fids_zf_OFF=f["fids_zf_OFF"][()],
                fids_zf_DIFF=f["fids_zf_DIFF"][()],
                ppm_array=f["ppm_array"][()],
                ppm_array_zf=f["ppm_array_zf"][()],
                n=int(f["n"][()]),
                spectral_width=float(f["spectral_width"][()]),
                Bo=float(f["Bo"][()]),
                dwell_time=float(f["dwell_time"][()]),
                te=float(f["te"][()]),
                vendor=f["vendor"][()].decode("utf-8"),
                metab_names_OFF=[s.decode("utf-8") for s in f["metab_names_OFF"][()]],
                MM_names_OFF=[s.decode("utf-8") for s in f["MM_names_OFF"][()]],
                metab_names_DIFF=[s.decode("utf-8") for s in f["metab_names_DIFF"][()]],
                MM_names_DIFF=[s.decode("utf-8") for s in f["MM_names_DIFF"][()]],
            )
