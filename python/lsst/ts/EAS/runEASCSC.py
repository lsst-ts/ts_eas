#!/usr/bin/env python3.6
import argparse

from EASCSC import EASCsc

parser = argparse.ArgumentParser(description='usage: runEASCSC (index)')
parser.add_argument('index', metavar='N', type=int, nargs=1, help='index number of the CSC implementation')
args = parser.parse_args()

component = EASCsc(args.index[0])
