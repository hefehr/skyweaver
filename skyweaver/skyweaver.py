from __future__ import annotations

import logging
import h5py
import numpy as np

from typing import Self, Any
from collections.abc import Iterable
from dataclasses import dataclass

from rich.progress import track
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.units import Quantity
from katpoint import Target, Antenna
from mosaic.beamforming import (
    DelayPolynomial,
    PsfSim,
    generate_nbeams_tiling,
    Tiling, BeamShape)


class DelayEngine:
    """A class for generating delay/weight solutions for beamforming
    """

    def __init__(self, subarray: Subarray, phase_reference: Target) -> None:
        """Create an instance of DelayEngine

        Args:
            subarray (Subarray): The superset of antennas that will be beamformed
            phase_reference (Target): The phase centre of the pointing
        """
        self.subarray = subarray
        self.phase_reference = phase_reference
        self._targets: list[tuple[Target, Subarray]] = []

    def _validate_subarray(self, subarray: Subarray) -> None:
        if subarray not in self.subarray:
            raise ValueError("Subarray is not a sub-set of full array")

    def add_tiling(
            self,
            tiling: Tiling,
            subarray: Subarray = None,
            prefix: str = None) -> None:
        """Add a tiling to the engine

        Args:
            tiling (Tiling): A mosaic Tiling object containing beam positions
            subarray (Subarray, optional): The subarray that should be used when beamforming.
                                           Defaults to the full array.
            prefix (str, optional): A prefix to be used when naming beams from this tiling.
                                    Defaults to the name of the tiling target.
        """
        if subarray is None:
            subarray = self.subarray
        else:
            self._validate_subarray(subarray)
        if prefix is None:
            prefix = tiling.beam_shape.bore_sight.name
        coordinates = tiling.get_equatorial_coordinates()
        for ii, (ra, dec) in enumerate(coordinates):
            self._targets.append(
                (Target(f"{prefix}{ii:07d},radec,{ra},{dec}"), subarray))

    def add_beam(self, target: Target, subarray: Subarray) -> None:
        """Add a single beam to the engine

        Args:
            target (Target): A katpoint target specifying the beam position
            subarray (Subarray): The subarray that should be used when beamforming.
                                 Defaults to the full array.
        """
        self._validate_subarray(subarray)
        self._targets.append((target, subarray))

    def _extract_weights(self) -> np.ndarray:
        weights = np.zeros(
            (len(
                self._targets),
                self.subarray.nantennas),
            dtype="float32")
        for ant_idx, antenna in enumerate(self.subarray.antenna_positions):
            for beam_idx, (target, subarray) in enumerate(self._targets):
                if antenna in subarray.antenna_positions:
                    weights[beam_idx, ant_idx] = 1.0
        return weights

    def calculate_delays(self, start: Time, end: Time,
                         step: TimeDelta) -> list[float, float, np.ndarray]:
        """Calculate weight/delay solutions for all specified beams
           in intervals over a time span.

        Args:
            start (Time): The start epoch
            end (Time): The end epoch
            step (TimeDelta): The time step between solutions

        Returns:
            list[float, float, np.ndarray]: A list of delay solutions with validity
                                            start times and durations.

        Notes:
            Delays are returned in the format:
            [(start epoch, validity duration in seconds, delay model)]

            The delay model itself is a numpy ndarray with dimensions
            (nbeams, nantennas, 3) where the inner 3-length dimension
            contains (scalar antenna weight, delay offset, delay rate).
        """
        targets = [i[0] for i in self._targets]
        weights = self._extract_weights().reshape(
            len(self._targets), self.subarray.nantennas, 1)
        delay_calc = DelayPolynomial(
            self.subarray.antenna_positions, self.phase_reference,
            targets, self.subarray.reference_antenna)
        models = []
        epochs = []
        current_time = start
        while current_time <= end:
            epochs.append(current_time)
            current_time += step
        for epoch in track(epochs, description="Generating delays..."):
            delays = delay_calc.get_delay_polynomials(
                epoch.to_datetime(),
                duration=step.to(u.s).value)
            # Delays are returned with shape (nbeam, nant, 2)
            # where the innermost dimension is (delays, rate)
            # Here we inject the beam weight into the delay array
            # to make the inner dimension (antenna weight, delay, rate).
            # The weight here refers to a scalar multiplier for the antenna
            # and is intended as a means to allow different subsets of antennas
            # to be used for beamforming.
            delays = np.concatenate((weights, delays), axis=-1)
            models.append((
                epoch.unix,
                step.to(u.s).value,
                delays))
        return models


@dataclass
class Subarray:
    # Mapping of antenna names to XEPhem metadata
    antenna_positions: list[Antenna]

    @property
    def reference_antenna(self) -> Antenna:
        return self.antenna_positions[0].array_reference_antenna()

    @property
    def nantennas(self) -> int:
        return len(self.antenna_positions)

    def __contains__(self, subarray: Subarray) -> bool:
        other_set: set = set(subarray.antenna_positions)
        self_set: set = set(self.antenna_positions)
        return other_set.issubset(self_set)

    def get_beam_shape(
            self,
            target: Target,
            epoch: Time,
            reference_frequency: Quantity) -> BeamShape:
        """Compute the synthesised beam shape for a given target.

        Args:
            target (Target): The target of observation
            epoch (Time): The time of observation
            reference_frequency (Quantity): The frequency at which to compute beam parameters

        Returns:
            mosaic.BeamShape: A mosaic beamshape object containing synthesised beam parameters
        """
        antenna_strings: list[str] = [ant.format_katcp()
                                      for ant in self.antenna_positions]
        psfsim: PsfSim = PsfSim(
            antenna_strings,
            reference_frequency.to(
                u.Hz).value)
        return psfsim.get_beam_shape(target, epoch.unix)

    def make_circular_tiling(
            self,
            target: Target,
            epoch: Time,
            reference_frequency: Quantity,
            nbeams: int,
            overlap: float = 0.5) -> Tiling:
        """Create a circular tiling

        Args:
            target (Target): The target of observation
            epoch (Time): The time of observation
            reference_frequency (Quantity): The frequency at which to compute beam parameters
            nbeams (int): The number of beams to generate
            overlap (float, optional): The overlap between neighbouring beams in a tile row. Defaults to 0.5.

        Returns:
            Tiling: A mosaic tiling object containing beam positions
        """
        beam_shape: BeamShape = self.get_beam_shape(
            target, epoch, reference_frequency)
        tiling: Tiling = generate_nbeams_tiling(
            beam_shape, nbeams, 0.5,
            "variable_size", "circle",
            None, "equatorial")
        return tiling


@dataclass
class ObservationMetadata:
    # Mapping of antenna names to XEPhem metadata
    antenna_positions: dict[str, Antenna]
    # Mapping of antenna names to ordinal indices in CBF stream
    antenna_feng_map: dict[str, int]
    # Observation centre frequency
    centre_frequency: Quantity
    # Observation bandwidth
    bandwidth: Quantity
    # ITRF reference position of the observatory
    itrf_reference: tuple[float, float, float]
    # Total number of frequency channels
    nchannels: int
    # The project identifier
    project_id: str
    # The schedule block identifier
    sb_id: str
    # The version of the CBF used for the observation
    cbf_version: str
    # The UNIX epoch at which the system was synchronised
    sync_epoch: Time
    # Time series of phase centres used during observation
    phase_centres: np.ndarray[str, str]
    # Time series of "suspect" flags indicating data validity
    # True values imply invalid data
    suspect_flags: np.ndarray[bool]

    @classmethod
    def from_file(cls, fname) -> Self:
        """Parse an HDf5 metdata file

        Args:
            fname (str): Path to the metadata file to be read

        Returns:
            Self: ObservationMetadata instance
        """
        with h5py.File(fname) as f:
            # Parse out the antenna positions
            antenna_positions: dict = dict()
            for antenna_descriptor in f["antenna_positions"]:
                kp_antenna: Antenna = Antenna(antenna_descriptor.decode())
                antenna_positions[kp_antenna.name] = kp_antenna
            # Parse out antenna to F-engine mapping
            antenna_feng_map: dict[str, int] = {
                antenna.decode(): index for antenna, index in f["antenna_feng_map"][()]}
            # Parse out general metadata
            metadata: dict = dict(f.attrs)
            return cls(
                antenna_positions,
                antenna_feng_map,
                metadata["cfreq"] * u.Hz,
                metadata["bandwidth"] * u.Hz,
                tuple(metadata["itrf_reference"]),
                metadata["nchans"],
                metadata["project_id"],
                metadata["sb_id"],
                metadata["cbf_version"],
                Time(metadata["sync_epoch"], format="unix"),
                f["phase_centres"][()].astype([
                    ("timestamp", "datetime64[us]"), ("value", "|S64")]),
                f["suspect_flags"][()].astype([
                    ("timestamp", "datetime64[us]"), ("value", "bool")])
            )

    def _drop_duplicate_values(self, ar: np.ndarray) -> np.ndarray:
        vals = ar["value"]
        idxs = np.concatenate(([0], np.where(vals[1:] != vals[:-1])[0] + 1))
        return ar[idxs]

    def _covert_to_windows(self, ar: np.ndarray,
                           sentinel_time: Time) -> list[tuple[Time, Time, Any]]:
        windows = []
        for ii in range(len(ar)):
            start, value = ar[ii]
            if ii + 1 == len(ar):
                end = sentinel_time
            else:
                end, _ = ar[ii + 1]
            windows.append((start, end, value))
        return windows

    def get_subarray(self, antenna_names: list[str]) -> Subarray:
        return Subarray([self.antenna_positions[name]
                        for name in antenna_names])

    def find_observing_windows(
        self, min_duration: Quantity = 60 * u.s, allow_suspect: bool = False
    ) -> list[tuple[Time, Time, Target, bool]]:
        """Find observing windows corresponding to phase centre and suspect flags

        Args:
            min_duration  (Quantity, optional): Only return observing windows bigger than this length.
            allow_suspect (bool, optional): Also return observing windows with suspect data. Defaults to False.


        Returns:
            list: A list of observing epochs specifying the start, end, phase centre and suspect flag.

        Notes:
            The timing of the suspect flags to phase centre changes is unpredictable and so to be safe
            it is necessary to only return valid observing windows above a given size. Typically invalid
            windows are less than 1 second long, but we leave the default as 60 second under the expectation
            that observations will be considerably longer than this.
        """
        # Rather than handle some special case for the end of timestamp
        # sequences, we here define a sentinel time far away in the future.
        # This is chosen to be a year after the start of the observation
        # for saftey.
        sentinel_time = self.phase_centres[0]["timestamp"] + \
            np.timedelta64(365, 'D')
        phase_centre_windows = self._covert_to_windows(
            self._drop_duplicate_values(self.phase_centres),
            sentinel_time)
        suspect_windows = self._covert_to_windows(
            self._drop_duplicate_values(self.suspect_flags),
            sentinel_time)
        overlaps: list = []
        for pc_start, pc_end, target_descriptor in phase_centre_windows:
            target = Target(target_descriptor.decode())
            for flag_start, flag_end, is_suspect in suspect_windows:
                if not allow_suspect and is_suspect:
                    continue
                if pc_start < flag_end and pc_end > flag_start:
                    overlap_start = max(pc_start, flag_start)
                    overlap_end = min(pc_end, flag_end)
                    duration = (overlap_end - overlap_start) / \
                        np.timedelta64(1, 's')
                    if duration < min_duration.to(u.s).value:
                        continue
                    if overlap_end == sentinel_time:
                        # We convert sentinel time to None to remove any ambiguity
                        # a value of None implies "to the end of the session"
                        overlap_end = None
                    overlaps.append((
                        Time(overlap_start),
                        Time(overlap_end),
                        target,
                        is_suspect
                    ))
        return overlaps


def example(metadata_file):
    # Simple example of how to use this thing
    om = ObservationMetadata.from_file(metadata_file)
    subarray = om.get_subarray(("m000", "m002", "m004", "m006"))
    windows = om.find_observing_windows()
    for start, end, target, _ in windows:
        print(
            start,
            "--->",
            end,
            target.name,
            (end -
             start).to(
                u.s).value,
            "seconds")
    """
    e.g.
    2024-02-16T11:16:08.957000000 ---> 2024-02-16T11:21:04.865000000 J1644-4559 295.90 seconds
    2024-02-16T11:21:22.092000000 ---> 2024-02-16T11:26:15.682000000 J1644-4559_Offset1 293.59 seconds
    2024-02-16T11:26:34.536000000 ---> 2024-02-16T11:31:25.723000000 J1644-4559_Offset2 291.18 seconds
    2024-02-16T11:31:52.503000000 ---> 2024-02-16T12:01:51.651000000 M28 1799.14 seconds
    2024-02-16T12:03:06.117000000 ---> 2024-02-16T12:08:04.364000000 J0437-4715 298.24 seconds
    """
    # Choose the first window
    start, end, target, _ = windows[0]
    nbeams = 400
    overlap = 0.5
    tiling = subarray.make_circular_tiling(
        target, start, om.centre_frequency, nbeams, overlap)
    de = DelayEngine(subarray, target)
    # add the tiling to the delay engine for a new subarray
    subsubarray = om.get_subarray(("m000", "m002"))
    de.add_tiling(tiling, subsubarray)
    # add a sigle beam but using the whole available array
    de.add_beam(target, subarray)
    # Calculate the delays with a 4 second interval
    step = TimeDelta(4 * u.s)
    delays = de.calculate_delays(start, end, step)
    return delays


def main():
    """
    What does this thing actually do?

    1. Read the observation metadata file - Done
    2. Read some kind of beamformer configuration file
    4. From the beamformer configuration: - Done
        - generate tilings
        - generate beam pointings
        - generate antenna to beam maps
    5. Using the antenna database and the metadata: - Done
        - Generate all delays required for the full observation
          and write them to a (temporary?) file
        - Calculate the PSFs for each beam as a function of time
        - Generate a sky coverage map for the run
    6. Write a configuration header for the beamformer specifying the configuration
    7. Compile the beamformer for the specific configuration
    8. Excute the beamformer passing the weights file a pipeline configuration and the input data
    9. ...
    10. Profit


    Need to define:
    - Configuration file format (preferrably YAML)
    - Antenna database file format (preferrably YAML)
    - Delay format (preferrably some packed binary format)
    """
    pass
