import awkward as ak

class Trigger:
    
    def __init__(self, arrays):
        """Initialise"""
        print("---> Loading Trigger...")
        self.arrays = arrays

    def get_triggers(self, thres=10):
        """
        Get triggers from nested PEs/layer arrays.
        Returns arrays with trigger flags
        """
        print("---> Triggering")
        
        required_fields = ["PEs_per_layer_L_end", "PEs_per_layer_DS"]
        if not all(field in self.arrays.fields for field in required_fields):
            raise ValueError(f"Input arrays must contain {required_fields}")
            
        # Sum channels for each layer
        layer_sums_L_end = ak.sum(self.arrays['PEs_per_layer_L_end'], axis=-1)  # [events, layer]
        layer_sums_DS = ak.sum(self.arrays['PEs_per_layer_DS'], axis=-1)        # [events, layer]
        
        # Create trigger flags for each layer
        self.arrays['trig_L_end_layers'] = ak.values_astype(layer_sums_L_end > thres, 'int')  # [events, layer]
        self.arrays['trig_DS_layers'] = ak.values_astype(layer_sums_DS > thres, 'int')        # [events, layer]
        
        # Check if all layers triggered
        self.arrays['trig_L_end'] = ak.values_astype(ak.sum(self.arrays['trig_L_end_layers'], axis=-1) == 4, 'bool')
        self.arrays['trig_DS'] = ak.values_astype(ak.sum(self.arrays['trig_DS_layers'], axis=-1) == 4, 'bool')
        
        # Combined trigger requirement
        self.arrays['trig'] = (self.arrays['trig_L_end'] & self.arrays['trig_DS'])
    
        n_tot = len(self.arrays)
        n_trig = len(self.arrays[self.arrays['trig']])
        trig_frac = n_trig/n_tot
    
        print(f"-> {n_trig}/{n_tot} = {100*trig_frac:.2f}% events have triggers")

        return self.arrays

    def apply_triggers(self):
        """
        Apply triggers: exclude events with no trigger
        """
        return self.arrays[self.arrays['trig']]