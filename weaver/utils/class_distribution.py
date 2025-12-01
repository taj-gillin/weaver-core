#!/usr/bin/env python
"""
Utility to compute class distribution from root files.
"""

import argparse
import glob
import yaml
import os
import numpy as np
from collections import defaultdict


def load_sample_list(yaml_file):
    """Load file patterns from a YAML sample list.
    
    Args:
        yaml_file: Path to YAML file containing list of file patterns
        
    Returns:
        List of resolved file paths
    """
    with open(yaml_file, 'r') as f:
        sample_entries = yaml.safe_load(f)
    
    if not sample_entries:
        raise ValueError(f"Samples file {yaml_file} is empty")
    
    if not isinstance(sample_entries, list):
        raise ValueError(f"Samples file {yaml_file} must contain a list of file paths or patterns")
    
    resolved_files = []
    samples_dir = os.path.dirname(os.path.abspath(yaml_file))
    
    for entry in sample_entries:
        if isinstance(entry, str):
            pattern = entry
        elif isinstance(entry, dict) and len(entry) == 1:
            pattern = list(entry.keys())[0]
        else:
            continue
        
        # Expand environment variables and user home directory
        pattern = os.path.expandvars(os.path.expanduser(pattern))
        
        # Make absolute path if relative
        if not os.path.isabs(pattern):
            pattern = os.path.abspath(os.path.join(samples_dir, pattern))
        
        # Glob to find matching files
        matched = glob.glob(pattern)
        if matched:
            resolved_files.extend(matched)
        else:
            print(f"WARNING: Pattern {pattern} matched no files")
    
    resolved_files = sorted(set(resolved_files))
    
    if not resolved_files:
        raise ValueError(f"No files found using samples file {yaml_file}")
    
    return resolved_files


def compute_class_distribution(file_paths, treename='Events', 
                               class_branches=None, verbose=True):
    """Compute class distribution from root files.
    
    Args:
        file_paths: List of root file paths or single YAML sample list
        treename: Name of TTree in root files (default: 'Events')
        class_branches: List of class label branch names. 
                       Default: ['recojet_isB', 'recojet_isC', 'recojet_isUDSG']
        verbose: Print progress information
        
    Returns:
        Dictionary with class counts and statistics
    """
    import uproot
    
    if class_branches is None:
        class_branches = ['recojet_isB', 'recojet_isC', 'recojet_isUDSG']
    
    # Handle YAML sample list
    if isinstance(file_paths, str):
        if file_paths.endswith('.yaml'):
            if verbose:
                print(f"Loading file patterns from {file_paths}")
            file_paths = load_sample_list(file_paths)
        else:
            file_paths = [file_paths]
    
    # Initialize counters
    class_counts = defaultdict(int)
    total_events = 0
    files_processed = 0
    
    if verbose:
        print(f"\nProcessing {len(file_paths)} files...")
        print(f"Class branches: {class_branches}\n")
    
    # Process each file
    for filepath in file_paths:
        try:
            with uproot.open(filepath) as f:
                # Get the tree
                if treename not in f:
                    if verbose:
                        print(f"WARNING: Tree '{treename}' not found in {filepath}, skipping")
                    continue
                
                tree = f[treename]
                
                # Read class labels
                labels = {}
                for branch in class_branches:
                    if branch in tree:
                        labels[branch] = tree[branch].array(library='np')
                    else:
                        if verbose:
                            print(f"WARNING: Branch '{branch}' not found in {filepath}, skipping file")
                        labels = None
                        break
                
                if labels is None:
                    continue
                
                # Count events for each class
                n_events = len(labels[class_branches[0]])
                total_events += n_events
                
                for branch in class_branches:
                    class_counts[branch] += np.sum(labels[branch])
                
                files_processed += 1
                
                if verbose:
                    print(f"  [{files_processed}/{len(file_paths)}] {os.path.basename(filepath)}: {n_events} events")
                    
        except Exception as e:
            if verbose:
                print(f"ERROR processing {filepath}: {e}")
            continue
    
    # Compute statistics
    results = {
        'total_events': total_events,
        'files_processed': files_processed,
        'class_counts': dict(class_counts),
        'class_fractions': {}
    }
    
    if total_events > 0:
        for branch, count in class_counts.items():
            results['class_fractions'][branch] = count / total_events
    
    return results


def print_distribution(results, class_names=None):
    """Pretty print the class distribution results.
    
    Args:
        results: Dictionary returned by compute_class_distribution
        class_names: Optional mapping from branch names to display names
    """
    if class_names is None:
        class_names = {
            'recojet_isB': 'B jets',
            'recojet_isC': 'C jets',
            'recojet_isUDSG': 'UDSG jets'
        }
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION SUMMARY")
    print("="*60)
    print(f"Total events: {results['total_events']:,}")
    print(f"Files processed: {results['files_processed']}")
    print("\n" + "-"*60)
    print(f"{'Class':<20} {'Count':>15} {'Fraction':>15}")
    print("-"*60)
    
    for branch, count in results['class_counts'].items():
        display_name = class_names.get(branch, branch)
        fraction = results['class_fractions'][branch]
        print(f"{display_name:<20} {count:>15,} {fraction:>14.2%}")
    
    print("="*60 + "\n")
    
    # Sanity check
    total_class_count = sum(results['class_counts'].values())
    if total_class_count != results['total_events']:
        print(f"WARNING: Sum of class counts ({total_class_count:,}) != total events ({results['total_events']:,})")
        print("This may indicate overlapping classes or missing labels.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compute class distribution from root files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From YAML sample list
  python class_distribution.py -i samples_testing.yaml
  
  # From individual root files
  python class_distribution.py -i file1.root file2.root file3.root
  
  # With custom tree name
  python class_distribution.py -i samples.yaml --treename MyTree
  
  # With custom class branches
  python class_distribution.py -i samples.yaml --classes label_b label_c label_udsg
        """
    )
    
    parser.add_argument('-i', '--input', required=True, nargs='+',
                       help='Input root files or YAML sample list')
    parser.add_argument('--treename', default='Events',
                       help='Name of TTree in root files (default: Events)')
    parser.add_argument('--classes', nargs='+',
                       default=['recojet_isB', 'recojet_isC', 'recojet_isUDSG'],
                       help='Class label branch names')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # If single input and it's a YAML file, use it directly
    if len(args.input) == 1 and args.input[0].endswith('.yaml'):
        input_files = args.input[0]
    else:
        input_files = args.input
    
    # Compute distribution
    results = compute_class_distribution(
        input_files,
        treename=args.treename,
        class_branches=args.classes,
        verbose=not args.quiet
    )
    
    # Print results
    print_distribution(results)
    
    return results


if __name__ == '__main__':
    main()
