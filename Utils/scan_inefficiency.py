import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

class ScanInefficiency:
    def __init__(self, start=10, stop=150, steps=29, verbose=True):
        """Initialise inefficiency scanner 
        Args:
            start (int): starting threshold
            stop (int): ending threshold
            steps (int): number of scan points
            verbose (bool): printouts 
        """
        print("---> Loading Inefficiency...")
        # Validate inputs
        if stop <= start:
            raise ValueError("'stop' must be greater than 'start'")
        if steps < 2:
            raise ValueError("'steps' must be >= 2")
        self.start = start
        self.stop = stop
        self.steps = steps
        self.n_layers = 4 # Number of layers/module
        # Create scan points
        self.thresholds = np.linspace(self.start, self.stop, self.steps)
        if verbose: 
            print(f"PE thresholds to scan:\n{self.thresholds}")        

    def get_ineff(self, k, N):
        """Calculate inefficiency
        
        Args:
            k (int): Number of failures
            N (int): Total number of triggers
        
        Returns:
            float: inefficiency (float)
        """
        return k / N if N > 0 else 0

    def get_ineff_err(self, k, N, conservative=True):
        """Calculate Wilson confidence interval for proportion
        
        Args:
            k (int): Number of failures
            N (int): Total number of trials
            conservative (bool): If True, use upper bound difference
        
        Returns:
            float: Uncertainty estimate
            
        Raises:
            ValueError: If k > N or if either is negative
        """
        if k > N or k < 0 or N < 0:
            raise ValueError("Invalid k or N values")
            
        lower, upper = proportion_confint(k, N, method="wilson")
        point = k/N
        
        if conservative:
            return abs((upper - point) / 2)
        else:
            return abs((lower - point) / 2)

    def _create_result_arrays(self):
        """Create fresh arrays for storing results"""
        ineff = {i: np.zeros(self.steps) for i in range(self.n_layers)}
        ineff_err = {i: np.zeros(self.steps) for i in range(self.n_layers)}
        return ineff, ineff_err
        
    def scan_ineff_arrays(self, arrays):
        """Scan array to calculate inefficiency at different PE thresholds
        
        Args:
            arrays: awkward array containing triggered and sorted PEs in test module
            
        Returns:
            tuple: (x, y, yerr)
                x: PE threshold values as list
                y: Inefficiency values as dict (keys are sorted layer indices)
                yerr: Inefficiency uncertainty as dict (keys are sorted layer indices)
        """
        # Get result arrays
        ineff, ineff_err = self._create_result_arrays()
        # Total triggers
        N = len(arrays)
        # For each scan point
        for i_thres, thres in enumerate(self.thresholds):
            # Mark failures 
            arrays[f"fail_{thres}"] = ak.values_astype(
                ( (arrays["PEs_per_layer_T_sorted"] <= thres) 
                 & (arrays["PEs_per_layer_T_sorted"] >= 0) # Edge case, I filter these but still 
                ), "int"
            ) 
            # For i/4 layers
            for i_layer in range(self.n_layers):
                # Get failures 
                k = ak.sum(arrays[f"fail_{thres}"][:,i_layer])
                # Get and store inefficiency
                ineff[i_layer][i_thres] = self.get_ineff(k, N)
                # Get and store uncertainty on inefficiency
                ineff_err[i_layer][i_thres] = self.get_ineff_err(k, N)
        return self.thresholds, ineff, ineff_err 

    def scan_ineff_hists(self, hists):
        """Scan histogram to calculate inefficiency at different PE thresholds
        
        Args:
            hists (dict): Sorted PE/layer histograms
            
        Returns:
            tuple: (x, y, yerr)
                x: PE threshold values as list
                y: Inefficiency values as dict (keys are sorted layer indices)
                yerr: Inefficiency uncertainty as dict (keys are sorted layer indices)
        """
        # Get result arrays
        ineff, ineff_err = self._create_result_arrays()

        # Get triggers from the 1/4 histogram
        N = np.sum(hists[self.n_layers-1]['counts'])
        
        for i_layer in range(self.n_layers):
        
            counts = hists[i_layer]['counts']
            bin_edges = hists[i_layer]['bin_edges']
            
            
            # For each scan point
            for i_thres, thres in enumerate(self.thresholds):
                # Find bins below threshold
                mask = bin_edges[:-1] < thres # Use left bin edge
                k = np.sum(counts[mask])
                
                # Calculate inefficiency
                ineff[i_layer][i_thres] = k / N if N > 0 else 0
                # Calculate uncertainty using instance method
                ineff_err[i_layer][i_thres] = self.get_ineff_err(k, N)
    
        return self.thresholds, ineff, ineff_err 

    def plot_scan(self, ineff, ineff_err, title=None, fout=None):
        """  
          Plot inefficiency scan 
        """  
        # Create figure and axes
        fig, ax = plt.subplots()

        # Loop through graphs and plot
        for i_layer in range(self.n_layers): 
            # Create this graph
            ax.errorbar(
              x=self.thresholds, y=ineff[i_layer], yerr=ineff_err[i_layer], label=f"{self.n_layers-i_layer}/4 layers",
              fmt='o', markersize=4, capsize=2, elinewidth=1
            )
        # Set log-y scale
        ax.set_yscale("log")

        # Titles
        ax.set_title(title)
        ax.set_ylabel("Inefficiency")
        ax.set_xlabel("PE threshold")

        # Legend
        ax.legend(loc="best")

        # Draw inefficiency line
        ax.axhline(y=1e-4, color='gray', linestyle='--')
        # ax.text(90, 0.33e-4, "99.99% efficiency", color="gray", fontsize="small") 

        # Draw
        plt.tight_layout()
    
        # Save 
        if fout:
          plt.savefig(fout, dpi=300, bbox_inches="tight")
          print("\n---> Wrote:\n\t", fout)
            
        # Show
        plt.show()