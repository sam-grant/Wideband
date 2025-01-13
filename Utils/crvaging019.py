import numpy as np
import awkward as ak

class CrvAging019: 
    """ Build PEs / layer for each module as 4x32 arrays per event """
    def __init__(self, arrays, sanity=True):
        """ Initialise """
        print("---> Loading CrvAging019...")
        self.arrays = arrays
        self.sanity = sanity # sanity checks
        self.n_layers = int(4)
        self.n_channels_per_layer = int(32)

    def check_layers(self, layers, reshape):
        """ Helper function to check reshaped data """ 
        # Compare with nested array layers
        for i_layer in range(self.n_layers):
            layer_match = np.array_equal(
                ak.to_numpy(layers[i_layer]),
                ak.to_numpy(reshape[:, i_layer, :])
            )
            if not layer_match:
                print(f"Layer {i_layer} mismatch!")
                print(f"Original layer {i_layer}:", ak.to_numpy(layers[i_layer])[0])
                print(f"Reshaped layer {i_layer}:", ak.to_numpy(T_reshape[:, i_layer ,:])[0])
            else:
                print(f"Layer {i_layer} matches ✓")
        
    def crv_L_end(self):
        """
        Map the L-end channels
        Produce PEs / layer array like [ [ [], [], [], [], ], [ [], [], [], [], ] ... ] events:layer:channel  
        FEBs 6-7 are split into two layers (channels 0-31 and 32-63).
        Final shape: (events, 4 layers, 32 channels)
    
        Layer mapping:
        - Layer 0: FEB6, channels 0-31
        - Layer 1: FEB6, channels 32-63
        - Layer 2: FEB7, channels 0-31
        - Layer 3: FEB7, channels 32-63
        """
        print("\n---> Mapping CRV-L-end")
        
        # Configuration for L-end module
        feb_indices = [6, 7]  # FEB indices 
        
        # Reshape
        l_end_reshape = ak.to_numpy(
            self.arrays['PEsTemperatureCorrected'][:, min(feb_indices):(max(feb_indices)+1), :] #  [:, 6:8, 0:64] layers 0-3
        ).reshape(-1, self.n_layers, self.n_channels_per_layer) # -1, 4, 64
        
        # Store 
        self.arrays['PEs_per_layer_L_end'] = l_end_reshape
    
        # Check against layers mapped by hand 
        if self.sanity:

            layers = [
                self.arrays['PEsTemperatureCorrected'][:,6,0:32], 
                self.arrays['PEsTemperatureCorrected'][:,6,32:64],
                self.arrays['PEsTemperatureCorrected'][:,7,0:32], 
                self.arrays['PEsTemperatureCorrected'][:,7,32:64] 
            ]
            
            self.check_layers(layers, l_end_reshape)

        return self.arrays

    def crv_T(self, single_ended=True):
        """
        Map the T channels, handling both single and double-ended readout.
        Produce PEs / layer array like [ [ [], [], [], [], ], [ [], [], [], [], ] ... ] events:layer:channel  
    
        For double-ended: sum of FEBs 2&4 and 3&5
        For single-ended: only FEBs 2&3
        
        Final shape: (events, 4 layers, 32 channels)
        
        Layer mapping:
        - Layer 0: FEB 2&4, channels 0-31
        - Layer 1: FEB 2&4, channels 32-63
        - Layer 2: FEB 3&5, channels 0-31
        - Layer 3: FEB 3&5, channels 32-63
        """
        print("\n---> Mapping CRV-T")
        
        # Initiliase numpy container for reshaped data
        T_reshape = [] 
        
        if single_ended:
        
            # Single-ended readout 
            feb_indices = [2, 3]  
            
            # Reshape
            T_reshape = ak.to_numpy(
                self.arrays['PEsTemperatureCorrected'][:, min(feb_indices):(max(feb_indices)+1), :] # [:, 2:6, 0:64]
            ).reshape(-1, self.n_layers, self.n_channels_per_layer)
        
        else: 
    
            # Double-ended readout
            feb_indices = [ (2,3), (4,5) ] 
    
            # Layers 0 & 1 
            T_01 = ak.to_numpy( 
                (self.arrays['PEsTemperatureCorrected'][:, feb_indices[0][0], :] + self.arrays['PEsTemperatureCorrected'][:, feb_indices[1][0], :]) # [:, 2, :] + [:, 4, :] layer 0 & 1
            ).reshape(-1, int(self.n_layers/2), self.n_channels_per_layer) # Shape: events:2:32
    
            # Layers 2 & 3 
            T_23 = ak.to_numpy( 
                (self.arrays['PEsTemperatureCorrected'][:, feb_indices[0][1], :] + self.arrays['PEsTemperatureCorrected'][:, feb_indices[1][1], :]) # [:, 3, :] + [:, 5, :] layer 2 & 3
            ).reshape(-1, int(self.n_layers/2), self.n_channels_per_layer) # Shape: events:2:32
    
            # Stack them along axis 1 to get final shape (events, 4, 32)
            T_reshape = np.concatenate([T_01, T_23], axis=1)

        # Store
        self.arrays['PEs_per_layer_T'] = T_reshape
 
        # Check against layers mapped by hand 
        if self.sanity:

            layers = [] 
            
            if single_ended:
                layers.append(self.arrays['PEsTemperatureCorrected'][:,2,0:32]) 
                layers.append(self.arrays['PEsTemperatureCorrected'][:,2,32:64])
                layers.append(self.arrays['PEsTemperatureCorrected'][:,3,0:32])
                layers.append(self.arrays['PEsTemperatureCorrected'][:,3,32:64]) 
            else: 
                layers.append((self.arrays['PEsTemperatureCorrected'][:,2,0:32] + self.arrays['PEsTemperatureCorrected'][:,4,0:32]))
                layers.append((self.arrays['PEsTemperatureCorrected'][:,2,32:64] + self.arrays['PEsTemperatureCorrected'][:,4,32:64]))
                layers.append((self.arrays['PEsTemperatureCorrected'][:,3,0:32] + self.arrays['PEsTemperatureCorrected'][:,5,0:32]))
                layers.append((self.arrays['PEsTemperatureCorrected'][:,3,32:64] + self.arrays['PEsTemperatureCorrected'][:,5,32:64]))

            self.check_layers(layers, T_reshape)

        return self.arrays

    def crv_DS(self):
        """
        Map the DS channels, this one is weird
        Produce PEs / layer array like [ [ [], [], [], [], ], [ [], [], [], [], ] ... ] events:layer:channel  
        
        Final shape: (events, 4 layers, 32 channels)
        
        Layer mapping:
        - Layer 0: FEB 0&1, channels 0-15
        - Layer 1: FEB 0&1, channels 16-31
        - Layer 2: FEB 0&1, channels 32-48
        - Layer 3: FEB 0&1, channels 49-64
        """
        print("\n---> Mapping CRV-DS")
        
        # Reshape
        DS_reshape = ak.to_numpy( 
            ak.concatenate([
                self.arrays['PEsTemperatureCorrected'][:,1,0:16], self.arrays['PEsTemperatureCorrected'][:,0,15::-1], # Layer 0 
                self.arrays['PEsTemperatureCorrected'][:,1,16:32], self.arrays['PEsTemperatureCorrected'][:,0,31:15:-1], # Layer 1
                self.arrays['PEsTemperatureCorrected'][:,1,32:48], self.arrays['PEsTemperatureCorrected'][:,0,47:31:-1], # Layer 2
                self.arrays['PEsTemperatureCorrected'][:,1,48:64], self.arrays['PEsTemperatureCorrected'][:,0,63:47:-1] # # Layer 3
            ], axis=-1)
        ).reshape(-1, self.n_layers, self.n_channels_per_layer)

        # Store
        self.arrays['PEs_per_layer_DS'] = DS_reshape
    
        # Check against layers mapped by hand 
        if self.sanity:
            
            layers = [
                ak.concatenate([self.arrays['PEsTemperatureCorrected'][:,1,0:16], self.arrays['PEsTemperatureCorrected'][:,0,15::-1]], axis=-1), 
                ak.concatenate([self.arrays['PEsTemperatureCorrected'][:,1,16:32], self.arrays['PEsTemperatureCorrected'][:,0,31:15:-1]], axis=-1),
                ak.concatenate([self.arrays['PEsTemperatureCorrected'][:,1,32:48], self.arrays['PEsTemperatureCorrected'][:,0,47:31:-1]], axis=-1),
                ak.concatenate([self.arrays['PEsTemperatureCorrected'][:,1,48:64], self.arrays['PEsTemperatureCorrected'][:,0,63:47:-1]], axis=-1)
            ]
            self.check_layers(layers, DS_reshape)

        return self.arrays