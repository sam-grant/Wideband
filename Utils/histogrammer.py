import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import pyplot 
pl = pyplot.Plot()

class Histogrammer:

    def __init__(self, nbins=100, x_range=(0,500)):
        """Initialise PEs / layer histogrammer"""
        print("---> Loading Hists...")
        self.nbins = nbins
        self.x_range = x_range
        self.n_layers = 4
        # Initiliase histograms
        self.hists = {}
        for i_sort in range(self.n_layers): # Iterate over layers
            # Create empty histogram with metadata
            self.hists[i_sort] = {
                'counts': np.zeros(self.nbins),
                'bin_edges': np.linspace(self.x_range[0], self.x_range[1], nbins + 1),
                'n_fills': 0
            }

    def acc_hists(self, arrays):
        """
        Accumulate histograms of sorted PEs/layer for the test module.
        Arrays: input data arrays.
        Histograms: existing histogram dictionary (if None, will be initialised)
        """
        if 'PEs_per_layer_T_sorted' not in arrays.fields:
            raise ValueError("Input arrays must contain 'PEs_per_layer_T_sorted'")
                    
        for i_sort in range(self.n_layers):
            # Get flattened array for this sort index
            array = ak.to_numpy(
                ak.flatten(arrays['PEs_per_layer_T_sorted'][:,i_sort], axis=None)
            )
            # Suppress zeros 
            # array = array[array > 0]
            
            # Create histogram data
            counts, _ = np.histogram(
                array, 
                bins=self.hists[i_sort]['bin_edges'],  # Use existing bin edges
            )
    
            # Accumulate counts
            self.hists[i_sort]['counts'] += counts
            self.hists[i_sort]['n_fills'] += 1
        
        return self.hists

    def plot_hists(self, hists, fout=None):

        fig, ax = plt.subplots(2, 2, figsize=(1.5*8, 1.5*6))

        for i_sort in range(len(hists)):
            
            # Convert i_sort to 2D index
            row = i_sort // 2
            col = i_sort % 2
        
            # Get histogram data
            counts = hists[i_sort]["counts"]
            bin_edges = hists[i_sort]["bin_edges"]
            bin_centres = (bin_edges[1:] + bin_edges[:-1])/2
            
            # Calculate stats
            n_entries = np.sum(counts)
            mean = np.sum(counts * bin_centres) / n_entries
            std_dev = np.sqrt(np.sum(counts * (bin_centres - mean)**2) / n_entries)
            
            # Plot histogram using bin edges and counts
            hist = ax[row,col].hist(
                x=hists[i_sort]["bin_edges"][:-1],  # Use left edges of bins
                bins=hists[i_sort]["bin_edges"],
                weights=hists[i_sort]["counts"],
                histtype="step",
                color="blue",
                log=True,
                label=f"Entries: {int(n_entries)}\nMean: {pl.RoundToSigFig(mean, 3)}\nStd Dev: {pl.RoundToSigFig(std_dev, 3)}"
            )
        
            ax[row,col].set_title(i_sort)
            ax[row,col].set_xlabel("PEs / layer")
            ax[row,col].set_ylabel("Events" if col == 0 else "")
            # ax[row,col].grid(True, alpha=0.7)
            ax[row,col].legend(loc="upper right")
            
        plt.tight_layout()
        if fout:
            plt.savefig(fout, dpi=300, bbox_inches="tight")
            print("\n---> Wrote:\n\t", fout)
        plt.show()
