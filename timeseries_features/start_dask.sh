#!/bin/bash

nohup dask-scheduler &
condor_submit dask.sub
