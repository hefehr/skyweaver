# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import pytest
from katpoint import Antenna
from skyweaver import Subarray


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
