import awkward as ak
import numpy as np

class Sort:
    """PE layer sorter"""
    def __init__(self):
        """Initialise"""
        print("---> Loading Sort...")
        
    def sort_layer_PEs(self, array):
        """
        Sort PEs per layer within each event.
        """ 
        # Sum channels for each layer 
        layer_sums = ak.sum(array, axis=-1)  # Shape: [events, layer]
        
        # Sort layer sums within each event (ascending order)
        sorted_layers = ak.sort(layer_sums, axis=-1)
        
        return sorted_layers

    def print_sorted_layer_PEs(self, array, n_events=10):
        print("Sorted PEs:")
        for i_event in range(n_events): 
            print(i_event, array[i_event])
        print("...")