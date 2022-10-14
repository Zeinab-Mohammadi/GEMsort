# GEMsort

This package provides flexible and easy-to-use code for Graph nEtwork Multichannel (GEMsort), a spike sorting approach for sorting hundred of spikes in real-time. The GEMsort code include options for sorting a Neuropixels dataset and a synthetic multichannel data. The code to produce the synthetic data is presented as well. 

GEMsort neural spike sorting algorithm can rapidly sort neural spikes without requiring significant computer processing power and system memory storage. The parallel processing architecture of GEMsort is particularly suitable for digital hardware implementation to improve processing speed and recording channel scalability. Multichannel synthetic neural spikes and actual neural recordings with Neuropixels probes were used to evaluate the sorting accuracies of the GEMsort algorithm.

Two signal processing modifications to our previously developed single-channel real-time spike sorting EGNG (Enhanced Growing Neural Gas) algorithm, which is largely based on graph network. Duplicated neural spikes were eliminated and represented by the neural spike with the strongest signal profile, significantly reducing the amount of neural data to be processed. In addition, the channel from which the representing neural spike was recorded was used as an additional feature to differentiate between neural spikes recorded from different neurons having similar temporal features. 

## Package Contents

### GEMsort
- Data preparation.py: Prepare and download the data
- Sorting_Neuropixels_data: Using GEMsort to extrct, filter and sort the spikes for real neural data
- Sorting_synthetic_data: Sorting the spikes of a multichannel synthetic data
- Making_synthetic_data: Producing synthetic data with different firing rates and spike shapes
- ...
- ...


## Installation

Please consider that this code needs python 3.9 to run and it is useful to see the env.yml file for more details about the configuration of this package.

To download and install the code, you ca download it using the link above or open a terminal window and type below command:

```
git clone https://github.com/Zeinab-Mohammadi/GEMsort.git
```

It would be useful to make a new virtual environment and then run the code. This will prevent any possible issue with other projects or version control problems.


## Code author

Zeinab Mohammadi

Please cite as:

Mohammadi, Z., Denman, D., Klug, A., & Lei, T. C. (2022). Multichannel neural spike sorting with spike reduction and positional feature. bioRxiv. 
