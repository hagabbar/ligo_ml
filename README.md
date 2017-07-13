# ligo_ml

* Train/Store a nonexistant neural network model:

  run neural_net.py -e 1000 -dd data/L1-DARMBLRMS-1168649431-30134.npy -da data/L1-ML-1168649431-30134.npy -o /home/hunter.gabbard/public_html/Detchar/scattering/scrambling/training_sets -p 0.6 -c june16-2017_scattering/data/june16/H1-SUS-DAMP_channels.txt -bs 7000 -t False --learning-rate 0.2  

* Test over an exisitng neural network model example:

  run neural_net.py -e 2000 -dd june16-2017_scattering/calibration_removed/june16/H1-DARMBLRMS-1181641722-49313.npy -da june16-2017_scattering/data/1Hz/june16/*ML* -o /home/hunter.gabbard/public_html/Detchar/scattering/june16-17/testing_set -p 0.5 -c june16-2017_scattering/data/june16/H1-SUS-DAMP_channels.txt -bs 7000 -t True --learning-rate 0.2 -m /home/hunter.gabbard/public_html/Detchar/scattering/june16-17/training_sets/best_run/nn_model.hdf -od june16-2017_scattering/calibration_removed/june16/H1-DARMBLRMS-1181641722-49313.npy.
