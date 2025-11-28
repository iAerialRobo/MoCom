# MoCom: Motion-based Inter-MAV Visual Communication Using Event Vision and Spiking Neural Networks

## Authors： Nengbo Zhang, Hann Woei Ho*, and Ye Zhou

## Introduction 

````
Reliable communication in Micro Air Vehicle (MAV) swarms is challenging in environments, where conventional radio-based methods suffer from spectrum congestion, jamming, and high power consumption. Inspired by the waggle dance of honeybees, which efficiently communicate the location of food sources without sound or contact, we propose a novel visual communication framework for MAV swarms using motion-based signaling. In this framework, MAVs convey information, such as heading and distance, through deliberate flight patterns, which are passively captured by event cameras and interpreted using a predefined visual codebook of four motion primitives: vertical (up/down), horizontal (left/right), left-to-up-to-right, and left-to-down-to-right, representing control symbols (``start'', ``end'', ``1'', ``0''). To decode these signals, we design an event frame-based segmentation model and a lightweight Spiking Neural Network (SNN) for action recognition. An integrated decoding algorithm then combines segmentation and classification to robustly interpret MAV motion sequences. Experimental results validate the framework's effectiveness, which demonstrates accurate decoding and low power consumption, and highlights its potential as an energy-efficient alternative for MAV communication in constrained environments.
````


# Citation
If you use the dataset and codes in an academic context, please cite our work:
````
Nengbo Zhang, Hann Woei Ho*, Ye Zhoue, MoCom: Motion-based Inter-MAV Visual Communication Using Event Vision and Spiking Neural Networks

(The academic paper was submitted to IEEE Transactions on Robotics)
````
