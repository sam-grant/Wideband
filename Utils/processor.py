import uproot
import awkward as ak
from concurrent.futures import ThreadPoolExecutor, as_completed

class Processor:
    """Class to handle reading and parallel processing """
    
    def __init__(self, reader, treenames=['run']):
        """Initialise the processor
        
        Args:
            reader: DataReader instance from anapytools
            treenames: List of tree names to read
        """
        print("---> Loading Processor...")
        self.treenames = treenames
        self.reader = reader

    def get_filelist(self, defname, run_number, quiet=False):
        """Get and filter file list
        
        Args:
            defname: Default name pattern for files
            run: run number (e.g., "002101")
            
        Returns:
            Filtered list of files
        """
        filelist = self.reader.get_file_list(defname=defname, quiet=True)
        filelist = [f for f in filelist if run_number in f]
        if not quiet: 
            print(f"filelist: {filelist}")
        return [f for f in filelist if run_number in f]
        
    def process_single_file(self, filename):
        """Process a single file and return its arrays
        
        Args:
            filename: Path to rec file
            
        Returns:
            Combined awkward array
        """
        # Open the file
        file = self.reader.read_file(filename, quiet=True)

        # Get trees
        trees_ = {treename: file[treename]
                 for treename in self.treenames
                 if treename in file}
        
        # Combine arrays
        arrays = []

        # run
        # ['runNumber', 'subrunNumber', 'timestamp', 'febID', 'spillsTotal', 'spillsRecorded', 'eventsRecorded', 'febSpills', 'febTemperaturesAvg', 'supplyMonitorsAvg', 'biasVoltagesAvg', 'pipeline', 'samples', 'PEs', 'PEsTemperatureCorrected', 'FWHMs', 'FWHMsTemperatureCorrected', 'signals', 'signalsTemperatureCorrected', 'chi2s', 'chi2sTemperatureCorrected', 'errors', 'errorsTemperatureCorrected', 'meanTemperatures', 'stddevTemperatures', 'maxedOutFraction', 'pedestals', 'calibConstants', 'calibConstantsTemperatureCorrected', 'noiseRate', 'xtalkProbability']
        # runSummary
        # ['runNumber', 'subrunNumber', 'spillIndex', 'spillNumber', 'boardStatus', 'FPGABlocks', 'spillTimestamp', 'eventNumber', 'tdcSinceSpill', 'timeSinceSpill', 'fitStatus', 'PEs', 'PEsTemperatureCorrected', 'temperature', 'pulseHeight', 'beta', 'time', 'LEtime', 'adc', 'recoStartBin', 'recoEndBin', 'pedestal', 'fitStatusReflectedPulse', 'PEsReflectedPulse', 'PEsTemperatureCorrectedReflectedPulse', 'pulseHeightReflectedPulse', 'betaReflectedPulse', 'timeReflectedPulse', 'LEtimeReflectedPulse', 'recoStartBinReflectedPulse', 'recoEndBinReflectedPulse', 'trackSlope', 'trackIntercept', 'trackChi2', 'trackPoints', 'trackPEs']
        # splills
        # ['runNumber', 'subrunNumber', 'spill_index', 'spill_num', 'spill_nevents', 'spill_neventsActual', 'spill_stored', 'spill_number_of_febs', 'spill_channels_per_feb', 'spill_number_of_samples', 'spill_biasVoltage', 'spill_temperature', 'spill_boardStatus', 'spill_FPGABlocks', 'spill_timestamp', 'spill_timestamp_sec', 'spill_timestamp_min', 'spill_timestamp_hour', 'spill_timestamp_mday', 'spill_timestamp_mon', 'spill_timestamp_year', 'spill_timestamp_wday', 'spill_timestamp_yday', 'spill_timestamp_isdst']

        for treename, tree in trees_.items():
            # Extract branches into an Awkward Array
            array = tree.arrays(
                filter_name=["runNumber", "subrunNumber", "eventNumber", "spillNumber", "spillIndex", "PEsTemperatureCorrected"],
                library="ak"
            )
            
            arrays.append(array)
        
        # Concatenate all arrays into a single Awkward Array
        arrays = ak.concatenate(arrays) if arrays else ak.Array([])

        # # Add subrun field
        # subrun = int(filename.split('_')[-1].replace('.root', ''))
        # subruns = [subrun] * len(arrays)
        # arrays = ak.with_field(arrays, subruns, "subrun")
        
        # Close file
        file.close()
        
        return arrays
    
    def process_files_parallel(self, filelist, max_workers=None):
        """Process multiple files in parallel
        
        Args:
            filelist: List of files to process
            max_workers: maximum number of worker threads (defaults to len(filelist))
            
        Returns:
            List of combined arrays from all files
        """
        if max_workers is None:
            max_workers = len(filelist)
        
        print(f"\n---> Starting parallel processing with {max_workers} workers...")
        print(f"Processing {len(filelist)} files in total\n")
        
        results = [] # Keep track of arrays in a list
        completed_files = 0
        total_files = len(filelist)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary mapping futures to their filenames
            futures = {
                executor.submit(self.process_single_file, file): file 
                for file in filelist
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    result = future.result()
                    results.append(result)  # Add to list
                    completed_files += 1
                    percent_complete = (completed_files / total_files) * 100
                    
                    # Extract just the base filename for cleaner output
                    base_filename = filename.split('/')[-1]
                    print(f"✓ {base_filename} processed successfully!")
                    print(f"  → Events in this file: {len(result)}")
                    print(f"  → Progress: {completed_files}/{total_files} files ({percent_complete:.1f}%)\n")
                    
                except Exception as e:
                    print(f"\n❌ Error processing {filename}:")
                    print(f"  → Error details: {str(e)}\n")

        # Concatenate all arrays at the end
        if results:
            arrays = ak.concatenate(results)
            total_events = len(arrays)
            print('\n---> Parallel processing completed!')
            print(f'---> Processed {completed_files}/{total_files} files successfully')
            print(f'---> Total events processed: {total_events:,}\n')
            return arrays
            
        return None
