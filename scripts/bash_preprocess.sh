#!/bin/bash
# cd /home/xxxxxxxx/multimodal_brain/src/tasks/run
for subject in {1..10}
do
  python preprocess/process_eeg_whiten.py --subject $subject
done