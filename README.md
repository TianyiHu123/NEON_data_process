# NEON_data_process
This is a repository for neon data processing using state-space model and PyMC to estimate SR flux posterior

## Overview
This repository contains the code using pdf estimation to estimate neon SR flux distribution.

## Input data
Input data is NEON site level hourly SR flux from 5 plot.
/${sitecode}/${sitecode}_${YYYY}_hourly_gC_allpos.csv

## example_code
Test-AR_Model_Updated.ipynb: original jupyternote book file
Example.py: clean example code extracted from Test-AR_Model_Updated.ipynb

## main
data_io.py: Read in the NEON site data
state_space_model.py: Construct state space model and use pymc to generate posterior

## ~
Data_lode.py: test data_io
Data_process.py: python script to process neon data on HPC, need to run in parallel to accelerate pymc


