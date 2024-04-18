# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import pytest
import numpy as np
from katpoint import Antenna
from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity
from skyweaver import Subarray, DelayModel


ANTENNAS = {
    'm000': 'm000,  -30:42:39.8,  21:26:38.0,  1035.0,  13.5,  -8.254 -207.2925 1.209 5875.794 5877.025,  -0:00:39.7 0 -0:04:04.4 -0:04:53.0 0:00:57.8 -0:00:13.9 0:13:45.2 0:00:59.8,  1.22',
    'm001': 'm001,  -30:42:39.8,  21:26:38.0,  1035.0,  13.5,  1.1275 -171.7635 1.0565 5869.964 5870.974,  -0:42:08.0 0 0:01:44.0 0:01:11.9 -0:00:14.0 -0:00:21.0 -0:36:13.1 0:01:36.2,  1.22',
    'm002': 'm002,  -30:42:39.8,  21:26:38.0,  1035.0,  13.5,  -32.1045 -224.2375 1.2545 5871.47 5872.221,  0:40:20.2 0 -0:02:41.9 -0:03:46.8 0:00:09.4 -0:00:01.1 0:03:04.7,  1.22',
    'm003': 'm003,  -30:42:39.8,  21:26:38.0,  1035.0,  13.5,  -66.5125 -202.2765 0.8885 5872.781 5874.412,  0:16:25.4 0 0:00:53.5 -0:02:40.6 0:00:00.9 0:00:05.7 0:34:05.6 0:02:12.3,  1.22'
    }

@pytest.fixture(name="example_subarray")
def fixture_example_subarray():
    # Create an example Subarray instance for testing
    antennas = [Antenna(ANTENNAS[key]) for key in ["m000", "m001", "m002"]]
    return Subarray(antenna_positions=antennas)

def test_reference_antenna(example_subarray):
    # Test the reference_antenna property
    reference_antenna = example_subarray.reference_antenna
    assert isinstance(reference_antenna, Antenna)
    assert reference_antenna == Antenna(
        ANTENNAS['m000']).array_reference_antenna()  # First antenna in the list

def test_nantennas(example_subarray):
    # Test the nantennas property
    assert example_subarray.nantennas == 3

def test_names(example_subarray):
    # Test the names property
    assert example_subarray.names == ['m000', 'm001', 'm002']

def test_contains_true(example_subarray):
    # Test __contains__ method for a valid subarray
    subarray_subset = Subarray(antenna_positions=[
        Antenna(ANTENNAS["m000"]), Antenna(ANTENNAS["m001"])])
    assert subarray_subset in example_subarray

def test_contains_false(example_subarray):
    # Test __contains__ method for an invalid subarray
    subarray_not_subset = Subarray(antenna_positions=[Antenna(ANTENNAS["m003"])])
    assert subarray_not_subset not in example_subarray

# Delay model tests






class TestDelayModel:

    # can create a DelayModel instance with valid inputs
    def test_create_delay_model_valid_inputs(self):
        start_epoch = Time("2022-01-01T00:00:00", format="isot", scale="utc")
        end_epoch = Time("2022-01-02T00:00:00", format="isot", scale="utc")
        delays = np.zeros((10, 5, 3))
        delay_model = DelayModel(start_epoch, end_epoch, delays)
    
        assert delay_model.start_epoch == start_epoch
        assert delay_model.end_epoch == end_epoch
        assert delay_model.delays.shape == (10, 5, 3)

    # can get the number of beams in the model
    def test_get_number_of_beams(self):
        start_epoch = Time("2022-01-01T00:00:00", format="isot", scale="utc")
        end_epoch = Time("2022-01-02T00:00:00", format="isot", scale="utc")
        delays = np.zeros((10, 5, 3))
        delay_model = DelayModel(start_epoch, end_epoch, delays)

        assert delay_model.nbeams == 10

    # can get the number of antennas in the model
    def test_get_number_of_antennas(self):
        start_epoch = Time("2022-01-01T00:00:00", format="isot", scale="utc")
        end_epoch = Time("2022-01-02T00:00:00", format="isot", scale="utc")
        delays = np.zeros((10, 5, 3))
        delay_model = DelayModel(start_epoch, end_epoch, delays)

        assert delay_model.nantennas == 5

    # can pack the delay model into bytes
    def test_pack_delay_model_into_bytes(self):
        start_epoch = Time("2022-01-01T00:00:00", format="isot", scale="utc")
        end_epoch = Time("2022-01-02T00:00:00", format="isot", scale="utc")
        delays = np.zeros((10, 5, 3))
        delay_model = DelayModel(start_epoch, end_epoch, delays)

        packed_bytes = delay_model.to_bytes()

        assert isinstance(packed_bytes, bytes)
        assert len(packed_bytes) > 0

    # can validate the epoch for the delay model
    def test_validate_epoch(self):
        start_epoch = Time("2022-01-01T00:00:00", format="isot", scale="utc")
        end_epoch = Time("2022-01-02T00:00:00", format="isot", scale="utc")
        delays = np.zeros((10, 5, 3))
        delay_model = DelayModel(start_epoch, end_epoch, delays)

        # Test with epoch within the validity window
        epoch_within_window = Time("2022-01-01T12:00:00", format="isot", scale="utc")
        assert delay_model.validate_epoch(epoch_within_window) is True

        # Test with epoch before the validity window
        epoch_before_window = Time("2021-12-31T12:00:00", format="isot", scale="utc")
        assert delay_model.validate_epoch(epoch_before_window) is False

        # Test with epoch after the validity window
        epoch_after_window = Time("2022-01-02T12:00:00", format="isot", scale="utc")
        assert delay_model.validate_epoch(epoch_after_window) is False

    # can get the phase for a given frequency and epoch
    def test_get_phase_for_frequency_and_epoch(self):
        # Create a delay model
        start_epoch = Time("2022-01-01T00:00:00", format="isot", scale="utc")
        end_epoch = Time("2022-01-02T00:00:00", format="isot", scale="utc")
        delays = np.zeros((10, 5, 3))
        delay_model = DelayModel(start_epoch, end_epoch, delays)

        # Define test inputs
        frequencies = Quantity([1e6, 2e6, 3e6], unit=u.Hz)
        epoch = Time("2022-01-01T12:00:00", format="isot", scale="utc")

        # Call the method under test
        phases = delay_model.get_phase(frequencies, epoch)

        # Perform assertions
        assert phases.shape == (10, 5, 3)
        assert np.allclose(phases, np.zeros((10, 5, 3)))

        # can get the phase for a single frequency
    def test_get_phase_single_frequency(self):
        start_epoch = Time("2022-01-01T00:00:00", format="isot", scale="utc")
        end_epoch = Time("2022-01-02T00:00:00", format="isot", scale="utc")
        delays = np.zeros((10, 5, 3))
        delay_model = DelayModel(start_epoch, end_epoch, delays)

        frequencies = Quantity([1e9], unit=u.Hz)
        epoch = Time("2022-01-01T12:00:00", format="isot", scale="utc")

        phases = delay_model.get_phase(frequencies, epoch)

        assert phases.shape == (10, 5, 1)
        assert np.allclose(phases, np.zeros((10, 5, 1)))