import awkward as ak 

class Clean:
    def __init__(self, arrays):
        """Initialise"""
        print("---> Loading Clean...")
        self.arrays = arrays
            
    def remove_negative_PEs(self): 
        """Remove events containing negative PE values
        Args:
            arrays: awkward array containing PE measurements 
        Returns:
            arrays: filtered array with only events containing non-negative PE values
        """
        
        has_negatives = ak.any(self.arrays["PEsTemperatureCorrected"] < 0, axis=-1)
        # Reset to event level 
        has_negatives_in_event = ak.any(has_negatives, axis=-1, keepdims=False) == True 
        # Count events
        n_events_negative = ak.sum(ak.values_astype(has_negatives, "int"))
        
        print(f"Removing {n_events_negative}/{len(self.arrays)} events containing negative PE values")
        
        # Remove events with negative PEs
        return self.arrays[~has_negatives_in_event]