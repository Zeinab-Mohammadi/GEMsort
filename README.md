# GEMsort

This package provides flexible and easy-to-use code for Graph nEtwork Multichannel (GEMsort), a spike sorting approach for sorting hundred of spikes in real-time. The GEMsort code includes options for sorting a Neuropixels dataset and synthetic multichannel data. The code to produce the synthetic data is included as well. 

GEMsort can rapidly sort neural spikes without requiring significant computer processing power and system memory storage. The parallel processing architecture of GEMsort is particularly suitable for digital hardware implementation to improve processing speed and recording channel scalability. Multichannel synthetic spikes and actual recordings with Neuropixels probes were used to evaluate the sorting accuracies of the GEMsort algorithm.

Two signal processing modifications were added to our single-channel real-time spike sorting EGNG (Enhanced Growing Neural Gas) algorithm, which is largely based on a graph network. Duplicated neural spikes were eliminated and represented by the spike with the strongest signal profile. This will significantly reduce the amount of neural data to be processed. In addition, the channel from which the representing spike was recorded was used as an additional feature to differentiate between neural spikes recorded from different neurons having similar temporal features. 

## Package Contents

### GEMsort
- Data preparation.py: Prepare and download the data
- Sorting_Neuropixels_data: Using GEMsort to extract, filter, and sort spikes of the real data
- Sorting_synthetic_data: Spike sorting of a multichannel synthetic data
- Making_synthetic_data: Producing synthetic data with different firing rates and spikes shapes


## Installation

Please consider that this code needs python 3.9 to run appropriately. It would be useful to make a new virtual environment and then run the code. This will prevent any possible issues with other projects or version control problems.

To download and install the code, you can download it using the link above or open a terminal window and type the below command:

```
git clone https://github.com/Zeinab-Mohammadi/GEMsort.git
```

## Code author

Zeinab Mohammadi

Please cite as:

Mohammadi, Zeinab, Daniel Denman, Achim Klug, and Tim C. Lei. "Multichannel neural spike sorting with spike reduction and positional feature." bioRxiv (2022). 
