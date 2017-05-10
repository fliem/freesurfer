#!/usr/bin/env python
import argparse
import os
import shutil
import nibabel
from glob import glob
from subprocess import Popen, PIPE
from shutil import rmtree
import subprocess
from warnings import warn
from qc import create_subject_plots, create_subject_report, create_group_report


def run(command, env={}, ignore_errors=False):
    merged_env = os.environ
    merged_env.update(env)
    # DEBUG env triggers freesurfer to produce gigabytes of files
    merged_env.pop('DEBUG', None)
    process = Popen(command, stdout=PIPE, stderr=subprocess.STDOUT, shell=True, env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0 and not ignore_errors:
        raise Exception("Non zero return code: %d"%process.returncode)

__version__ = open('/version').read()

parser = argparse.ArgumentParser(description='FreeSurfer quality control reports.')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the'
                    'participant level analysis.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                    'Multiple participant level analyses can be run independently '
                    '(in parallel) using the same output_dir.',
                    choices=['participant', 'group'])
parser.add_argument('--participant_label', help='The label of the participant that should be analyzed. The label '
                   'corresponds to sub-<participant_label> from the BIDS spec '
                   '(so it does not include "sub-"). If this parameter is not '
                   'provided all subjects should be analyzed. Multiple '
                   'participants can be specified with a space separated list.',
                   nargs="+")

parser.add_argument('-v', '--version', action='version',
                    version='BIDS-App example version {}'.format(__version__))

args = parser.parse_args()


if not args.participant_label:
    subject_dirs = glob(os.path.join(args.output_dir, "sub-*"))
    fsid_list = [os.path.basename(subject_dir) for subject_dir in subject_dirs]
else:
    fsid_list = ["sub-" + s for s in args.participant_label]

# workaround for https://mail.nmr.mgh.harvard.edu/pipermail//freesurfer/2016-July/046538.html
output_dir = os.path.abspath(args.output_dir)
qc_dir = os.path.join(output_dir, "00_qc")

# running participant level
if args.analysis_level == "participant":
    for fsid in fsid_list:
        print("Creating report for %s." % fsid)
        create_subject_plots(output_dir, fsid)
        create_subject_report(qc_dir, fsid)

elif args.analysis_level == "group":
    print("Creating group report.")
    create_group_report(qc_dir, fsid_list)