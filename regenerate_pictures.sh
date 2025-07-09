#!/bin/bash

python3 -m src.plot.PlotMetrics --dataset_name cifar10 --inner_iterations 50 --batch_size_alignement 512
python3 -m src.plot.PlotMetrics --dataset_name mnist --inner_iterations 50 --batch_size_alignement 512
python3 -m src.plot.PlotMetrics --dataset_name ixi --batch_size_alignement 16
python3 -m src.plot.PlotMetrics --dataset_name heart_disease --batch_size_alignement 16
python3 -m src.plot.PlotMetrics --dataset_name synth --inner_iterations 1 --batch_size_alignement 1
python3 -m src.plot.PlotMetrics --dataset_name synth_complex --inner_iterations 1 --batch_size_alignement 1