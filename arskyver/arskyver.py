#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sigpyproc.header import Header
from sigpyproc.readers import FilReader
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import glob
import os
import sys
import time
from subprocess import Popen, PIPE, STDOUT, run
import traceback


def read_dada_header(dada_file):
    with open(dada_file, "rb") as f:
        header = f.read(4096)
        dada_header = {}
        for line in header.decode().split("\n"):
            entries = line.split()
            if len(entries) == 2:
                try:
                    dada_header[entries[0]] = float(entries[1])
                except ValueError:
                    dada_header[entries[0]] = entries[1]
    return dada_header


class Beam(object):

    def __init__(
        self,
        parfile,
        name,
        pos,
        header,
        cachedir,
        outputdir,
        tscrunch,
        stokes_mode,
        subint_len,
        nbins,
    ):

        self.parfile = parfile
        self.predfile = parfile.replace(".par", ".pred")
        self.name = name
        self.pos = pos
        self.cachedir = cachedir
        self.outputdir = outputdir
        self.subint_len = subint_len
        self.stokes_mode = stokes_mode
        self.tscrunch = tscrunch
        self.nbins = nbins

        self.dada_header = header
        self.parse_header()

        self.data = []
        self.filcount = 0
        self.nsamp_total = 0
        self.nsamp = 0

        self.dspsr_proc = None

        with open(self.parfile, "r") as pf:
            lines = pf.readlines()
            for line in lines:
                if len(line.split()) == 0:
                    continue
                if line.split()[0] == "DM":
                    self.DM = line.split()[1]

        try:
            os.mkdir(f"{self.cachedir}/{self.name}")
        except FileExistsError:
            pass

    def parse_header(self):

        self.tsamp = (
            int(self.dada_header["OBS_NCHAN"])
            / self.dada_header["OBS_BW"]
            * self.tscrunch
        )
        self.chan0_idx = int(self.dada_header["CHAN0_IDX"])

        self.fBW_lo = (
            self.dada_header["OBS_FREQ"]
            + self.dada_header["OBS_BW"] / int(self.dada_header["OBS_NCHAN"]) / 2
            - self.dada_header["OBS_BW"] / 2
        )

        self.fBW_hi = self.fBW_lo + self.dada_header["OBS_BW"]

        self.foff = -self.dada_header["OBS_BW"] / int(self.dada_header["OBS_NCHAN"])
        self.fhi = self.fBW_lo + (
            int(self.dada_header["CHAN0_IDX"]) + int(self.dada_header["NCHAN"]) - 1
        ) * self.dada_header["BW"] / int(self.dada_header["NCHAN"])

        self.freq_str = int(
            self.fhi + (int(self.dada_header["NCHAN"] / 2) - 0.5) * self.foff
        )

        self.nchan = int(self.dada_header["NCHAN"])
        self.npol = len(self.stokes_mode)
        self.t0 = Time(self.dada_header["MJD_START"], scale="utc", format="mjd")
        self.timestamp_str = self.t0.datetime.strftime("%Y-%m-%d-%H:%M:%S")

    def make_t2pred(self):

        cmd = [
            "tempo2",
            "-npsr",
            "1",
            "-f",
            self.parfile,
            "-pred",
            f"meerkat {self.t0.mjd} {(self.t0 + 1 * u.d).mjd} {self.fBW_lo/1e6} {self.fBW_hi/1e6} 12 2 3599.9999999999998",
        ]

        print(" ".join(cmd))
        run(cmd)

        os.rename("t2pred.dat", self.name + ".pred")

    def __call__(self, tfb_data):

        self.data.append(tfb_data[:, ::-1, :].transpose(0, 2, 1))

        self.nsamp += len(self.data[-1])
        current_subint_len = self.nsamp * self.tsamp

        if current_subint_len >= self.subint_len:
            self.flush()

    def flush(self):

        if self.dspsr_proc is not None:
            self.dspsr_proc.wait()
            if self.dspsr_proc.returncode != 0:
                raise ValueError("Error occured when folding with DSPSR")
            self.delete_fil()

        print(
            "ARSKYVER: Writing",
            f"{self.cachedir}/{self.name}/{self.name}_{self.filcount:05d}.fil",
            self.nsamp,
            flush=True,
        )
        self.write_fil()
        self.fold_fil()
        self.nsamp_total += self.nsamp
        self.nsamp = 0
        self.data = []

    def write_fil(self):

        tstart = (self.t0 + self.nsamp_total * self.tsamp * u.s).mjd

        H = Header(
            filename=f"{self.cachedir}/{self.name}/{self.name}_{self.filcount:05d}.fil",
            data_type="raw data",
            nchans=self.nchan,
            foff=self.foff / 1e6,
            fch1=self.fhi / 1e6,
            nbits=8,
            frame="topocentric",
            tsamp=self.tsamp,
            tstart=tstart,
            nsamples=self.nsamp,
            backend="FAKE",
            coord=self.pos,
            nifs=self.npol,
            telescope="MeerKAT",
        )

        FW = H.prep_outfile(
            f"{self.cachedir}/{self.name}/{self.name}_{self.filcount:05d}.fil",
            rescale=False,
        )

        for f in range(len(self.data)):
            FW.write(self.data[f].tobytes())

    def fold_fil(self):

        # fmt: off
        cmd = [
            "dspsr",
            "-set", "state=Stokes",
            "-b", str(self.nbins),
            "-P",self.name + ".pred",
            "-D",self.DM,
            "-O",f"{self.cachedir}/{self.name}/{self.name}_{self.filcount:05d}",
            f"{self.cachedir}/{self.name}/{self.name}_{self.filcount:05d}.fil",
        ]
        # fmt: on

        print(f"ARSKYVER: running {' '.join(cmd)}", flush=True)
        self.dspsr_proc = Popen(cmd)

    def delete_fil(self):

        os.remove(f"{self.cachedir}/{self.name}/{self.name}_{self.filcount:05d}.fil")
        self.filcount += 1

    def psradd(self):

        if not os.path.exists(self.outputdir + f"/{self.name}"):
            os.mkdir(self.outputdir + f"/{self.name}")

        # For some reason, psradd only takes the first set of predictors when combining the subint archives.
        # This set only spans 1 hour, so causes errors for later subints.
        # Adding -E <parfile> re-applies the timing model when combining, to fix this problem.
        # fmt: off
        cmd = [
            "psradd",
            "-E", self.parfile,
            "-o", f"{self.outputdir}/{self.name}/{self.name}_{self.freq_str}.ar",
        ]
        # fmt: on

        print(
            f"ARSKYVER: running {' '.join(cmd)} {self.cachedir}/{self.name}/{self.name}_?????.ar",
            flush=True,
        )

        for ar in range(self.filcount):
            cmd.append(f"{self.cachedir}/{self.name}/{self.name}_{ar:05d}.ar")

        run(cmd)

        # for ar in range(self.filcount):
        #     os.remove(f"{self.cachedir}/{self.name}/{self.name}_{ar:05d}.ar")


class ArSkyVer(object):

    def __init__(
        self,
        input_file_list,
        cachedir,
        outputdir,
        delay_file,
        coherent_DM,
        stokes_mode,
        subintlen,
        tscrunch,
        nbins,
    ):

        self.outputdir = outputdir
        self.delay_file = delay_file
        self.coherent_DM = coherent_DM
        self.stokes_mode = stokes_mode
        self.tscrunch = tscrunch

        # Open voltages DADA file to get observation parameters
        self.input_file_list = input_file_list
        with open(self.input_file_list, "r") as ifl:
            files = ifl.readlines()
        self.dada_header = read_dada_header(files[0].strip())
        self.parse_header()

        # Setup the cache directory for this process
        self.cache_dir = cachedir

        self.skyweaver_outputdir = (
            self.cache_dir + f"/{self.timestamp_str}/0/{self.freq_str}"
        )

        if os.path.exists(self.skyweaver_outputdir):
            if len(glob.glob(self.skyweaver_outputdir + "/*tfb")) != 0:
                raise ValueError(
                    f"Error, TFB files already present in cache dir, {self.skyweaver_outputdir}"
                )
        else:
            os.makedirs(self.skyweaver_outputdir)

        # Identify which beams to fold, and setup T2 predictor files
        with open(self.delay_file + ".targets", "r") as dat:
            targets = dat.readlines()

        self.parfiles = []
        self.predfiles = []
        self.beampos = []
        self.beamnames = []
        self.beamidx = []
        bidx = 0
        for target in targets:
            if target.split(",")[0] == "name":
                continue

            name = target.split(",")[0]
            raj = target.split(",")[2].strip()
            decj = target.split(",")[3].strip()

            if os.path.exists(outputdir + "/" + name + ".par"):
                self.parfiles.append(outputdir + "/" + name + ".par")
                self.predfiles.append(outputdir + "/" + name + ".pred")
                self.beamnames.append(name)
                self.beampos.append(SkyCoord(raj, decj, unit="hourangle,deg"))
                self.beamidx.append(bidx)

            else:
                print(
                    f"Warning, no parfile or predfile for beam {name}. This beam will be deleted."
                )

            bidx += 1

        self.beams = []
        for b, pf in enumerate(self.parfiles):
            root = outputdir + f"/{self.beamnames[b]}"
            self.beams.append(
                Beam(
                    pf,
                    self.beamnames[b],
                    self.beampos[b],
                    self.dada_header,
                    self.skyweaver_outputdir,
                    outputdir,
                    tscrunch,
                    stokes_mode,
                    subintlen,
                    nbins,
                )
            )

        self.nbeams = len(self.beams)

    def make_t2preds(self):

        for beam in self.beams:
            beam.make_t2pred()

    def parse_header(self):

        self.tsamp = int(self.dada_header["OBS_NCHAN"]) / self.dada_header["OBS_BW"]
        self.chan0_idx = int(self.dada_header["CHAN0_IDX"])

        self.fBW_lo = (
            self.dada_header["OBS_FREQ"]
            + 0.5 * self.dada_header["OBS_BW"] / int(self.dada_header["OBS_NCHAN"])
            - self.dada_header["OBS_BW"] / 2.0
        )

        self.fBW_hi = self.fBW_lo + self.dada_header["OBS_BW"]

        self.foff = -self.dada_header["OBS_BW"] / int(self.dada_header["OBS_NCHAN"])
        self.fhi = self.fBW_lo + (
            int(self.dada_header["CHAN0_IDX"]) + int(self.dada_header["NCHAN"]) - 1
        ) * self.dada_header["BW"] / int(self.dada_header["NCHAN"])

        self.freq_str = int(
            self.fhi + (int(self.dada_header["NCHAN"] / 2) - 0.5) * self.foff
        )

        self.nchan = int(self.dada_header["NCHAN"])
        self.npol = int(self.dada_header["NPOL"])
        self.t0 = Time(self.dada_header["MJD_START"], scale="utc", format="mjd")
        self.timestamp_str = self.t0.datetime.strftime("%Y-%m-%d-%H:%M:%S")

    def launch_skyweaver(self):

        # fmt: off
        cmd = [
            "skyweavercpp",
            "--input-file", self.input_file_list,
            "--delay-file", self.delay_file,
            "--output-dir", self.cache_dir,
            "--enable-incoherent-dedispersion=0",
            "--ddplan", str(self.coherent_DM),
            "--stokes-mode",self.stokes_mode,
            "--gulp-size", "65536",
            "--output-level", "6",
            "--statistics", "0",
            "--write-incoherent-beam", "0",
        ]
        # fmt: on

        self.skyweaver_process = Popen(cmd)

    def beamform_and_fold(self):

        self.launch_skyweaver()

        no_files_cnt = 0
        expected_obs_offset = 0
        while True:

            g = sorted(glob.glob(self.skyweaver_outputdir + "/*.tfb"))

            if len(g) == 0:
                if self.skyweaver_process.poll() is not None:
                    for beam in self.beams:
                        beam.flush()

                    for beam in self.beams:
                        beam.dspsr_proc.wait()
                        if beam.dspsr_proc.returncode != 0:
                            raise ValueError("Error occured when folding with DSPSR")
                        beam.delete_fil()
                        beam.psradd()

                    return self.skyweaver_process.returncode
                else:
                    no_files_cnt += 1
                    if no_files_cnt % 10 == 0:
                        print(f"No new files for {no_files_cnt} seconds")
                    time.sleep(1)
            else:
                no_files_cnt = 0

            for tfb in g:
                with open(tfb, "rb") as dat:

                    header = dat.read(4096)
                    for line in header.decode().split("\n"):
                        if len(line.split()) == 2:
                            if line.split()[0] == "OBS_OFFSET":
                                obs_offset = int(line.split()[1])
                            if line.split()[0] == "NBEAM":
                                nb = int(line.split()[1])

                    if obs_offset != expected_obs_offset:
                        raise ValueError("Error: .tfb files are not contiguous")

                    D = np.frombuffer(dat.read(), dtype="int8").reshape(
                        -1, int(self.dada_header["NCHAN"]), nb, len(self.stokes_mode)
                    )

                    expected_obs_offset += D.size

                    for b in range(self.nbeams):
                        fildat = np.asarray(
                            D[:, :, self.beamidx[b], :] + 128.0, dtype="uint8"
                        )
                        self.beams[b](fildat)

                os.remove(tfb)
                print(f"ARSKYVER: Processed {tfb}", flush=True)


if __name__ == "__main__":

    from optparse import OptionParser

    desc = """Run skyweaver, monitoring the output path for new tfb files, convert these to per-beam filterbanks, and fold with DSPSR"""

    parser = OptionParser(usage=" %prog [options]", description=desc)
    parser.add_option(
        "-i",
        "--input_file_list",
        type=str,
        default=None,
        help="List of DADA files containing voltages for skyweaver beamforming",
    )
    parser.add_option(
        "-c",
        "--cachedir",
        type=str,
        default=None,
        help="Base path for cache (ideally on a big ramdisk) where .tfb and .fil files will be produced and processed. A sub-directory named <cachedir>/<timestamp>/0/<frequency> will be produced.",
    )

    parser.add_option(
        "-o",
        "--outputdir",
        type=str,
        default=None,
        help="Base directory for output files. Parfiles for each beam should be placed in here. Subfolders with archives for each beam will be produced here.",
    )

    parser.add_option(
        "-s", "--subintlen", type=float, help="Desired subint length (in seconds)."
    )
    parser.add_option(
        "-d",
        "--delay_file",
        type=str,
        help="Delays file for skyweaver. Beam names will be obtained from <delay_file>.targets, and any beam which has a corresponding .par file at <outputdir>/<beamname>.par will be folded.",
    )

    parser.add_option(
        "-D",
        "--coherent_DM",
        type=float,
        help="DM used for coherent de-dispersion within skyweaver",
    )
    parser.add_option(
        "-S", "--stokes_mode", type=str, default="IQUV", help="Stokes mode"
    )
    parser.add_option(
        "-P",
        "--make_t2preds",
        action="store_true",
        default=False,
        help="Don't actually run the beamforming, just generate tempo2 predictors for DSPSR folding. These are required because otherwise the folding is massively slowed down by repeated tempo2 calls. This is not done automatically before beamforming, as we generally have many parallel beamforming jobs and race conditions could occur when moving t2pred.dat -> <beamname>.dat",
    )
    parser.add_option("-T", "--Tscrunch", type=int, help="Skyweaver Tscrunch factor")
    parser.add_option(
        "-b", "--nbins", type=int, default=256, help="Number of bins in folded profile"
    )

    options, args = parser.parse_args()

    f = ArSkyVer(
        options.input_file_list,
        options.cachedir,
        options.outputdir,
        options.delay_file,
        options.coherent_DM,
        options.stokes_mode,
        options.subintlen,
        options.Tscrunch,
        options.nbins,
    )

    if options.make_t2preds:
        f.make_t2preds()
        sys.exit(0)

    for b in range(len(f.beams)):
        if not os.path.exists(f.predfiles[b]):
            print(
                f"Error: .pred file for beam {f.beamnames[b]} not found in {options.outputdir}.\nEnsure .par files exist there, with names matching those in {options.delay_file}.targets,\nand re-run with -P to generate .pred files."
            )
            sys.exit(1)

    try:
        f.beamform_and_fold()
    except:
        f.skyweaver_process.kill()
        print(traceback.format_exc())
        sys.exit(1)
