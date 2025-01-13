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
        for treename, tree in trees_.items():
            # Extract branches into an Awkward Array
            array = tree.arrays(
                filter_name=["eventNumber", "PEsTemperatureCorrected"],
                library="ak"
            )
            
            arrays.append(array)
        
        # Concatenate all arrays into a single Awkward Array
        arrays = ak.concatenate(arrays) if arrays else ak.Array([])

        # Add subrun field
        subrun = int(filename.split('_')[-1].replace('.root', ''))
        subruns = [subrun] * len(arrays)
        arrays = ak.with_field(arrays, subruns, "subrun")
        
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
