#!/usr/bin/env python3
"""
Script to parse the most recent .eval file and move PDFs marked as "I" to files/linguistic/usable
"""
import os
import shutil
import glob
from inspect_ai.analysis import samples_df
from pathlib import Path

def get_most_recent_eval_file(logs_dir="logs"):
    """Find the most recent .eval file in the logs directory"""
    eval_files = glob.glob(os.path.join(logs_dir, "*.eval"))
    if not eval_files:
        raise FileNotFoundError("No .eval files found in logs directory")
    
    # Sort by modification time, newest first
    eval_files.sort(key=os.path.getmtime, reverse=True)
    return eval_files[0]

def move_pdfs_marked_interesting(samples, source_dir="files/linguistic", target_dir="files/linguistic/usable"):
    """Move PDFs marked as 'I' from source to target directory"""
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    moved_files = []
    
    # Filter for samples marked as "I" (interesting)
    for sample in samples:
        # Check if this sample is marked as interesting
        # Look in the scores for the value 'I'
        is_interesting = False
        
        if 'scores' in sample:
            scores = sample['scores']
            for score_name, score_data in scores.items():
                if isinstance(score_data, dict) and score_data.get('value') == 'I':
                    is_interesting = True
                    break
        
        # Also check older format
        output = sample.get('output', '')
        target_value = sample.get('target', '')
        if output == 'I' or target_value == 'I':
            is_interesting = True
        
        if is_interesting:
            # Extract filename from the input or messages
            pdf_filename = None
            
            # Look in messages for document attachments
            if 'messages' in sample:
                for message in sample['messages']:
                    if 'content' in message:
                        content = message['content']
                        if isinstance(content, list):
                            for content_item in content:
                                if isinstance(content_item, dict):
                                    if content_item.get('type') == 'document' and 'filename' in content_item:
                                        filename = content_item['filename']
                                        if filename.endswith('.pdf'):
                                            pdf_filename = filename
                                            break
                    if pdf_filename:
                        break
            
            # Fallback: Try to extract filename from different possible fields
            if not pdf_filename:
                if 'input' in sample and isinstance(sample['input'], str):
                    if '.pdf' in sample['input']:
                        pdf_filename = sample['input'].split('/')[-1] if '/' in sample['input'] else sample['input']
                
                if 'file' in sample:
                    pdf_filename = sample['file']
                
                # Look for any field that might contain the PDF filename
                for key, value in sample.items():
                    if isinstance(value, str) and '.pdf' in value:
                        potential_filename = value
                        if '/' in potential_filename:
                            potential_filename = potential_filename.split('/')[-1]
                        if potential_filename.endswith('.pdf'):
                            pdf_filename = potential_filename
                            break
            
            if pdf_filename:
                source_path = os.path.join(source_dir, pdf_filename)
                target_path = os.path.join(target_dir, pdf_filename)
                
                if os.path.exists(source_path):
                    try:
                        shutil.move(source_path, target_path)
                        moved_files.append(pdf_filename)
                        print(f"Moved: {pdf_filename}")
                    except Exception as e:
                        print(f"Error moving {pdf_filename}: {e}")
                else:
                    print(f"File not found: {source_path}")
    
    return moved_files

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