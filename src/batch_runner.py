import os
import glob
import concurrent.futures
import time
from tqdm import tqdm  # <-- Import tqdm

# Import the main parser function from your svg_parser.py
from svg_parser import parse_svg_to_contract

def process_single_svg(svg_path, output_dir):
    """Worker function to process a single file and catch any individual crashes."""
    try:
        # The parser function automatically saves the JSON to the output_dir
        parse_svg_to_contract(
            svg_file_path=svg_path,
            output_dir=output_dir,
            epsilon=0.5,
            max_edges_per_node=3
        )
        return True, svg_path
    except Exception as e:
        return False, f"{svg_path} - Error: {e}"

def batch_process(input_folder, output_folder, max_workers=None, limit=None):
    """Processes SVGs in an input folder and saves to the output folder.
    
    Args:
        limit (int, optional): If set, only process this many files.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all SVG files in the target directory
    search_pattern = os.path.join(input_folder, "*.svg")
    svg_files = glob.glob(search_pattern)
    
    if not svg_files:
        print(f"⚠️ No SVG files found in '{input_folder}'. Skipping...")
        return

    # Apply the limit if one was provided
    if limit is not None:
        svg_files = svg_files[:limit]
        print(f"⚠️ TEST MODE: Limiting execution to the first {limit} files.")

    print(f"🚀 Found {len(svg_files)} files to process in '{input_folder}'. Starting...")
    
    success_count = 0
    fail_count = 0
    failed_files = []

    start_time = time.time()

    # Process in parallel using CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the process pool
        futures = {executor.submit(process_single_svg, svg, output_folder): svg for svg in svg_files}
        
        # Gather results as they finish, wrapped in tqdm for the progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(input_folder)}"):
            success, message = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append(message)
                # Use tqdm.write instead of print so it doesn't scramble the progress bar
                tqdm.write(f"❌ Failed: {message}")

    elapsed = time.time() - start_time
    print(f"\n✅ Finished processing '{input_folder}' in {elapsed:.2f} seconds.")
    print(f"   Success: {success_count} | Failed: {fail_count}")
    if failed_files:
        print("   Review failed files above.")

if __name__ == "__main__":
    # Define the input and output mappings
    directories_to_process = [
        {"in": "data/train", "out": "contracts/train"},
        {"in": "data/test",  "out": "contracts/test"}
    ]
    
    # ==========================================
    # SET YOUR TEST LIMIT HERE
    # Change to None when you are ready to run all files
    # ==========================================
    TEST_LIMIT = None
    
    # Run the batch job for both train and test folders
    for job in directories_to_process:
        print(f"\n{'='*60}")
        print(f"  Starting Queue: {job['in']}  ->  {job['out']}")
        print(f"{'='*60}")
        batch_process(job["in"], job["out"], limit=TEST_LIMIT)