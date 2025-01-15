import awkward as ak
from pyplot import Plot
pl = Plot()

class Cuts:
    """Handle cuts"""
   
    def __init__(self, arrays, plot=True, verbose=True):
        """Initialise with arrays and verbosity flag"""
        print("\n---> Loading Cuts...")
        self.arrays = ak.copy(arrays) # Make a copy so we can always plot the "before" distributions
        self.plot = plot
        self.verbose = verbose
        self.n_initial = len(arrays)
       
        if self.verbose:
            print(f"Initial number of events: {self.n_initial}")
   
    def tot_PE_cut(self, min_PEs=900, max_PEs=2250, fout=None):
        """Cut on total PEs across all FEBs to exclude EM activity"""
        n_before = len(self.arrays)
       
        # Sum over all FEBs
        # 
        # tot_PEs = ak.sum(
        #    ak.flatten(self.arrays['PEsTemperatureCorrected'], axis=-1),
        #    axis=-1
        # )

        def get_tot_PEs(): 
            
            # Sum over each module's layers
            tot_PEs = (
                ak.sum(ak.sum(self.arrays['PEs_per_layer_L_end'], axis=-1), axis=-1) +
                ak.sum(ak.sum(self.arrays['PEs_per_layer_T'], axis=-1), axis=-1) +
                ak.sum(ak.sum(self.arrays['PEs_per_layer_DS'], axis=-1), axis=-1)
            )

            return tot_PEs 

        tot_PEs = get_tot_PEs()
        # print(tot_PEs)
    
        # Apply cut
        self.arrays = self.arrays[(tot_PEs >= min_PEs) & (tot_PEs < max_PEs)]

        if self.plot: 
            tot_PEs_after = get_tot_PEs()
            
            pl.Plot1DOverlay(
                {"Before" : tot_PEs, "After" : tot_PEs_after}, 
                nbins=1000, xmin=0, xmax=10000, log_x=True, log_y=True,
                xlabel='Total PEs', ylabel='Events',
                fout=fout
                # ,x_lines=[max_PEs]
            )
            
        if self.verbose:
            n_after = len(self.arrays)
            n_removed = n_before - n_after
            print(f"\nTotal PE cut (< {max_PEs}):")
            print(f"Events removed: {n_removed} ({100*n_removed/n_before:.1f}%)")
            print(f"Events remaining: {n_after} ({100*n_after/self.n_initial:.1f}% of initial)")
           
        return self.arrays

    def fiducial_cut(self, lo_chan=12, hi_chan=19, fout=None):
        """Create fiducial area cut on top trigger module, excluding outer channels"""
        n_before = len(self.arrays)
           
        # Get array and create mask
        array_before = self.arrays["PEs_per_layer_L_end"]
       
        mask = ak.Array([
            [
                [True if lo_chan <= i_channel <= hi_chan else False for i_channel in range(32)]
                for i_layer in range(4)
            ]
            for ievent in range(len(array_before))
        ])
       
        # Apply mask
        array_after = ak.mask(array_before, mask)
        self.arrays["PEs_per_layer_L_end"] = array_after

        # Instead of masking, set values outside fiducial region to zero
        # for i_layer in range(4):
        #     array_after = ak.where(
        #         (ak.Array([[i for i in range(32)] for _ in range(len(array))]) >= lo_chan) & 
        #         (ak.Array([[i for i in range(32)] for _ in range(len(array))]) <= hi_chan),
        #         array,
        #         0.0
        #     )

        # self.arrays["PEs_per_layer_L_end"] = array_after

        # Create a single channel index array that will broadcast
        # channel_idx = ak.Array([[[i for i in range(32)] for _ in range(4)]])
        
        # # Create fiducial mask and apply in one operation
        # fiducial_mask = (channel_idx >= lo_chan) & (channel_idx <= hi_chan)
        # array_after = ak.where(fiducial_mask, array, 0.0)
        # self.arrays["PEs_per_layer_L_end"] = array_after # ak.where(fiducial_mask, array, 0.0)

        if self.plot: 
            # Create the channel map 
            channels_before = ak.Array([
                [[i_channel for i_channel in range(32)]
                for i_layer in range(4)]
                for ievent in range(len(array_before))
            ])
        
            # Apply mask to channel map
            channels_after = ak.mask(channels_before, mask)
            
            # Flatten both channels and weights consistently
            channels_before = ak.flatten(channels_before, axis=None)
            channels_after = ak.flatten(channels_after, axis=None)
            weights_before = ak.flatten(array_before, axis=None)
            weights_after = ak.flatten(array_after, axis=None)
            
            pl.Plot1DOverlay( 
                {
                "Before" : channels_before,
                "After" : channels_after
                },
                nbins = 32, xmin=0, xmax=32,
                weights = [ 
                    weights_before, 
                    weights_after
                ],
                xlabel="Channel % 32", ylabel="PEs", title="CRV-L-end",
                fout=fout
            )
            
        if self.verbose:
            n_after = len(self.arrays)
            n_removed = n_before - n_after
            print(f"\nFiducial cut (channels {lo_chan}-{hi_chan}):")
            print(f"Events removed: {n_removed} ({100*n_removed/n_before:.1f}%)")
            print(f"Events remaining: {n_after} ({100*n_after/self.n_initial:.1f}% of initial)")
        
        return self.arrays
   
    def counters_hit_cut(self, min_counter_hits=(2*8), max_counter_hits=2*((2*8)+(4)), fout=None):
        """Cut on total number of non-zero channels across modules"""
        n_before = len(self.arrays)
       
        def get_counter_hits(array=None):
            if array is None:
                array = self.arrays["PEsTemperatureCorrected"]
                nonzero_counters = array > 0
            # Sum over channels first, then FEBs (event:feb:channel)
            return ak.sum(ak.sum(nonzero_counters, axis=2), axis=1)
    
        counter_hits_before = get_counter_hits()        
       
        # Apply cut
        self.arrays = self.arrays[(counter_hits_before >= min_counter_hits) & (counter_hits_before < max_counter_hits)]
    
        counter_hits_after = get_counter_hits() 

        if self.plot: 
            pl.Plot1DOverlay( 
                {
                    "Before" : ak.flatten(counter_hits_before, axis=None),
                    "After" : ak.flatten(counter_hits_after, axis=None)
                },
                nbins = 150, xmin=0, xmax=150, log_y=True,
                xlabel="Counter hits", ylabel="Events",
                fout=fout
            )
            
        if self.verbose:
            n_after = len(self.arrays)
            n_removed = n_before - n_after
            print(f"\nCounters hit cut (≤ {max_counter_hits}):")
            print(f"Events removed: {n_removed} ({100*n_removed/n_before:.1f}%)")
            print(f"Events remaining: {n_after} ({100*n_after/self.n_initial:.1f}% of initial)")

        return self.arrays