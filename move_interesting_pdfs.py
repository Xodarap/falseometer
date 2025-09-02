#!/usr/bin/env python3
"""
Script to parse the most recent .eval file and move PDFs marked as "I" to files/linguistic/usable
"""
import os
import shutil
import glob
from inspect_ai.analysis import samples_df

def get_most_recent_eval_file(logs_dir="logs"):
    """Find the most recent .eval file in the logs directory"""
    eval_files = glob.glob(os.path.join(logs_dir, "*.eval"))
    if not eval_files:
        raise FileNotFoundError("No .eval files found in logs directory")
    
    # Sort by modification time, newest first
    eval_files.sort(key=os.path.getmtime, reverse=True)
    return eval_files[0]

def main():
    try:
        # Get the most recent eval file
        most_recent_eval = get_most_recent_eval_file()
        print(f"Processing most recent eval file: {most_recent_eval}")
        
        # Parse the eval file
        samples = samples_df(most_recent_eval)
        print(f"Loaded {len(samples)} samples from eval file")

        incorrect = samples[samples['score_model_graded_qa'] == 'I']
        source_paths = incorrect['metadata_paper'].tolist()
        for source_path in source_paths:
            filename = source_path.split('/')[-1]
            target_path = os.path.join("files/linguistic/usable", filename)
            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
                print(f"Moved: {filename}")
            else:
                print(f"File not found: {source_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()