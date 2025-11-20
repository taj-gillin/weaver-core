#!/usr/bin/env python3
"""
Compute standardization parameters from all training files and generate updated configs.
This script processes all output_*_train.root files and creates standardized configs for both pnet and part formats.
"""

import os
import sys
import glob
import numpy as np
import awkward as ak
import yaml
import uproot
from pathlib import Path
from tqdm import tqdm

# Add weaver-core to path
thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../..'))
sys.path.append(os.path.join(weavercoredir, 'weaver'))
sys.path.append(weavercoredir)

from weaver.utils.data.config import DataConfig
from weaver.utils.data.fileio import _read_files
from weaver.utils.data.preprocess import _apply_selection, _build_new_variables
from weaver.utils.data.tools import _get_variable_names

class StandardizationComputer:
    """Compute standardization parameters from all training files."""
    
    # Variables that use -9 as dummy value for neutral particles (track-based variables)
    TRACK_VARIABLES_WITH_DUMMY = {
        'btagJetDistSig', 'btagJetDistVal', 'btagSip2dSig', 'btagSip2dVal',
        'btagSip3dSig', 'btagSip3dVal', 'dxy', 'dz'
    }
    DUMMY_VALUE = -9.0
    
    def __init__(self, data_dir, config_template_pnet, config_template_part, 
                 sample_fraction=0.1, max_events_per_file=None, training_files=None,
                 data_source_desc=None):
        """
        Args:
            data_dir: Directory containing output_*_train.root files
            config_template_pnet: Path to ParticleNet config template
            config_template_part: Path to ParticleTransformer config template
            sample_fraction: Fraction of events to use from each file (default: 0.1 = 10%)
            max_events_per_file: Maximum events per file (None = use all)
        """
        self.data_dir = data_dir
        self.data_source_desc = data_source_desc or data_dir or "custom training file list"
        self.config_template_pnet = config_template_pnet
        self.config_template_part = config_template_part
        self.sample_fraction = sample_fraction
        self.max_events_per_file = max_events_per_file
        
        # Find all training files
        if training_files is not None:
            self.training_files = sorted(set(training_files))
            if len(self.training_files) == 0:
                raise ValueError("No training files found using the provided samples list")
        else:
            if data_dir is None:
                raise ValueError("Either data_dir or training_files must be provided")
            pattern = os.path.join(data_dir, 'output_*_train.root')
            self.training_files = sorted(glob.glob(pattern))
            if len(self.training_files) == 0:
                raise ValueError(f"No training files found matching pattern: {pattern}")
        
        print(f"Found {len(self.training_files)} training files from {self.data_source_desc}")
    
    def load_data_sample(self, data_config, file_list=None):
        """Load a sample of data from files."""
        if file_list is None:
            file_list = self.training_files
        
        print(f"\nLoading data from {len(file_list)} files...")
        print(f"Using {self.sample_fraction*100:.1f}% of events from each file")
        if self.max_events_per_file:
            print(f"Maximum {self.max_events_per_file} events per file")
        
        # Load branches needed for preprocessing
        all_input_vars = set()
        for var_list in data_config.input_dicts.values():
            all_input_vars.update(var_list)
        
        # Add variables needed for new_variables and selection
        load_branches = all_input_vars.copy()
        if data_config.selection:
            load_branches.update(_get_variable_names(data_config.selection))
        
        # Add variables needed for derived variables
        for var_name, expr in data_config.var_funcs.items():
            load_branches.update(_get_variable_names(expr))
        
        # Load data from all files
        all_tables = []
        for filepath in tqdm(file_list, desc="Loading files"):
            try:
                # Determine load range
                if self.max_events_per_file:
                    with uproot.open(filepath) as f:
                        treename = self._get_treename(f)
                        tree = f[treename]
                        total_events = tree.num_entries
                        # Use sample_fraction but cap at max_events_per_file
                        num_events = min(int(total_events * self.sample_fraction), self.max_events_per_file)
                        load_range = (0, num_events / total_events)
                else:
                    load_range = (0, self.sample_fraction)
                
                # Load data
                table = _read_files(
                    [filepath], 
                    load_branches, 
                    load_range=load_range,
                    show_progressbar=False,
                    treename=data_config.treename,
                    branch_magic=data_config.branch_magic,
                    file_magic=data_config.file_magic
                )
                
                if len(table) > 0:
                    all_tables.append(table)
                    print(f"  Loaded {len(table)} events from {os.path.basename(filepath)}")
            except Exception as e:
                print(f"  WARNING: Failed to load {filepath}: {e}")
                continue
        
        if len(all_tables) == 0:
            raise ValueError("No data loaded from any files!")
        
        # Concatenate all tables
        from weaver.utils.data.tools import _concat
        combined_table = _concat(all_tables)
        print(f"\nTotal events loaded: {len(combined_table)}")
        
        # Apply selection and build new variables
        if data_config.selection:
            combined_table = _apply_selection(combined_table, data_config.selection, funcs=data_config.var_funcs)
            print(f"Events after selection: {len(combined_table)}")
        
        combined_table = _build_new_variables(combined_table, data_config.var_funcs)
        
        return combined_table
    
    def _get_treename(self, uproot_file):
        """Get tree name from uproot file."""
        treenames = set([k.split(';')[0] for k, v in uproot_file.items() 
                        if getattr(v, 'classname', '') == 'TTree'])
        if len(treenames) == 1:
            return treenames.pop()
        elif len(treenames) == 0:
            raise RuntimeError("No TTree found in file")
        else:
            raise RuntimeError(f"Multiple trees found: {treenames}")
    
    def _is_track_variable_with_dummy(self, var_name):
        """Check if variable uses -9 as dummy value."""
        # Check if var_name matches any of the track variable names
        # (handles both 'dxy' and 'pfcand_dxy' formats)
        for track_var in self.TRACK_VARIABLES_WITH_DUMMY:
            # Exact match
            if var_name == track_var:
                return True
            # Match with underscore prefix (e.g., 'pfcand_dxy' matches 'dxy')
            if var_name.endswith('_' + track_var):
                return True
            # Match with underscore prefix and check it's not part of another word
            # (e.g., 'pfcand_btagJetDistSig' matches 'btagJetDistSig')
            if var_name.endswith(track_var) and len(var_name) > len(track_var):
                # Check that there's an underscore before the track_var
                idx = var_name.rfind(track_var)
                if idx > 0 and var_name[idx-1] == '_':
                    return True
        return False
    
    def compute_statistics(self, table, var_name):
        """Compute robust statistics for a variable."""
        try:
            if var_name not in table.fields:
                return None
            
            var_data = table[var_name]
            
            # Flatten if it's a jagged array
            if isinstance(var_data, ak.Array):
                flat_data = ak.flatten(var_data, axis=None)
                flat_data = ak.to_numpy(flat_data)
            else:
                flat_data = np.array(var_data)
            
            # Remove NaN and Inf
            flat_data = flat_data[np.isfinite(flat_data)]
            
            # Check if this is a track variable that uses -9 as dummy value
            is_track_var = self._is_track_variable_with_dummy(var_name)
            
            # Count dummy values for reporting
            dummy_count = 0
            if is_track_var:
                dummy_mask = np.isclose(flat_data, self.DUMMY_VALUE)
                dummy_count = np.sum(dummy_mask)
                # Exclude dummy values from statistics calculation
                # (but keep them in the data for transformation)
                flat_data_for_stats = flat_data[~dummy_mask]
            else:
                flat_data_for_stats = flat_data
            
            if len(flat_data) == 0:
                return None
            
            # Compute robust statistics using filtered data (excluding dummy values)
            if len(flat_data_for_stats) == 0:
                # All values are dummy values, can't compute meaningful stats
                print(f"  WARNING: {var_name} has no non-dummy values (all are {self.DUMMY_VALUE})")
                return None
            
            # Compute percentiles and statistics on non-dummy values
            low, center, high = np.percentile(flat_data_for_stats, [16, 50, 84])
            
            # Robust scale (similar to AutoStandardizer)
            scale = max(high - center, center - low)
            scale = 1.0 if scale == 0 else 1.0 / scale
            
            stats = {
                'name': var_name,
                'count': len(flat_data),  # Total count (including dummy values)
                'count_non_dummy': len(flat_data_for_stats),  # Count excluding dummy values
                'dummy_count': int(dummy_count),  # Count of dummy values
                'mean': float(np.mean(flat_data_for_stats)),  # Mean excluding dummy values
                'median': float(center),  # Median excluding dummy values
                'std': float(np.std(flat_data_for_stats)),  # Std excluding dummy values
                'min': float(np.min(flat_data)),  # Min including dummy values (for reporting)
                'max': float(np.max(flat_data)),  # Max including dummy values (for reporting)
                'percentile_16': float(low),  # Percentile excluding dummy values
                'percentile_84': float(high),  # Percentile excluding dummy values
                'center': float(center),  # Center for standardization (excluding dummy values)
                'scale': float(scale),  # Scale for standardization (excluding dummy values)
                'has_nan': np.any(np.isnan(ak.to_numpy(ak.flatten(table[var_name], axis=None)))),
                'has_inf': np.any(np.isinf(ak.to_numpy(ak.flatten(table[var_name], axis=None)))),
            }
            
            return stats
        except Exception as e:
            print(f"Error computing stats for {var_name}: {e}")
            return None
    
    def compute_all_statistics(self, table, data_config):
        """Compute statistics for all input variables."""
        print(f"\n{'='*60}")
        print("Computing Standardization Parameters")
        print(f"{'='*60}")
        
        # Get all variables used in inputs
        all_input_vars = set()
        for var_list in data_config.input_dicts.values():
            all_input_vars.update(var_list)
        
        stats_dict = {}
        print(f"\nComputing statistics for {len(all_input_vars)} variables...")
        
        for var_name in sorted(all_input_vars):
            stats = self.compute_statistics(table, var_name)
            if stats:
                stats_dict[var_name] = stats
                dummy_info = ""
                if stats.get('dummy_count', 0) > 0:
                    dummy_info = f" (excluded {stats['dummy_count']} dummy values)"
                print(f"  {var_name:30s} center={stats['center']:10.6f} scale={stats['scale']:10.6f}{dummy_info}")
            else:
                print(f"  {var_name:30s} FAILED to compute statistics")
        
        return stats_dict
    
    def update_config_with_standardization(self, config_path, stats_dict, output_path):
        """Update config file with standardization parameters."""
        print(f"\nUpdating config: {os.path.basename(config_path)}")
        
        # Ensure output_path is absolute
        output_path = os.path.abspath(output_path)
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update preprocess method
        if 'preprocess' not in config:
            config['preprocess'] = {}
        config['preprocess']['method'] = 'manual'
        
        # Update input variables with standardization parameters
        for input_group in config['inputs'].values():
            for var_entry in input_group['vars']:
                if isinstance(var_entry, list):
                    var_name = var_entry[0]
                    if var_name in stats_dict:
                        stats = stats_dict[var_name]
                        
                        # Ensure list has enough elements
                        while len(var_entry) < 6:
                            if len(var_entry) == 1:
                                var_entry.append(None)  # center
                            elif len(var_entry) == 2:
                                var_entry.append(1)     # scale
                            elif len(var_entry) == 3:
                                var_entry.append(-999999)  # clip_min (very large = no clipping)
                            elif len(var_entry) == 4:
                                var_entry.append(999999)   # clip_max (very large = no clipping)
                            elif len(var_entry) == 5:
                                var_entry.append(0)     # pad_value
                        
                        # Update with standardization parameters
                        var_entry[1] = stats['center']  # subtract_by (center)
                        var_entry[2] = stats['scale']   # multiply_by (scale)
                        
                        # Disable clipping by using very large values - let robust standardization handle outliers
                        # Note: np.clip requires actual numbers, so we use very large values to effectively disable clipping
                        var_entry[3] = -999999  # clip_min (effectively no clipping)
                        var_entry[4] = 999999    # clip_max (effectively no clipping)
                        
                        # Keep pad_value if it exists, otherwise default to 0
                        if var_entry[5] is None:
                            var_entry[5] = 0   # pad_value
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save updated config
        with open(output_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
        
        print(f"  Saved to: {output_path}")
    
    def generate_configs(self, output_dir):
        """Generate standardized configs for both pnet and part."""
        print(f"\n{'='*60}")
        print("Generating Standardized Configs")
        print(f"{'='*60}")
        
        # Load configs
        data_config_pnet = DataConfig.load(self.config_template_pnet)
        data_config_part = DataConfig.load(self.config_template_part)
        
        # Load data using pnet config (both use same variables)
        print("\nLoading data sample...")
        table = self.load_data_sample(data_config_pnet)
        
        # Compute statistics
        stats_dict_pnet = self.compute_all_statistics(table, data_config_pnet)
        
        # For part, we use the same statistics (same variables)
        stats_dict_part = stats_dict_pnet.copy()
        
        # Generate output configs - ensure absolute paths
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_pnet = os.path.abspath(os.path.join(output_dir, 'data_config_pnet_standardized.yaml'))
        output_part = os.path.abspath(os.path.join(output_dir, 'data_config_part_standardized.yaml'))
        
        self.update_config_with_standardization(
            self.config_template_pnet, stats_dict_pnet, output_pnet)
        self.update_config_with_standardization(
            self.config_template_part, stats_dict_part, output_part)
        
        # Save statistics report
        report_path = os.path.abspath(os.path.join(output_dir, 'standardization_report.txt'))
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Standardization Parameters Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Data source: {self.data_source_desc}\n")
            f.write(f"Training files: {len(self.training_files)}\n")
            f.write(f"Events used: {len(table)}\n")
            f.write(f"Sample fraction: {self.sample_fraction*100:.1f}%\n\n")
            
            f.write("Standardization Parameters:\n")
            f.write("-"*60 + "\n")
            for var_name, stats in sorted(stats_dict_pnet.items()):
                f.write(f"\n{var_name}:\n")
                if stats.get('dummy_count', 0) > 0:
                    f.write(f"  Dummy values (-9): {stats['dummy_count']} out of {stats['count']} total\n")
                    f.write(f"  Non-dummy count: {stats['count_non_dummy']}\n")
                    f.write(f"  NOTE: Statistics computed excluding dummy values\n")
                f.write(f"  Center (median): {stats['center']:.6f}\n")
                f.write(f"  Scale: {stats['scale']:.6f}\n")
                f.write(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}\n")
                f.write(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"  16th percentile: {stats['percentile_16']:.6f}\n")
                f.write(f"  84th percentile: {stats['percentile_84']:.6f}\n")
                if stats['has_nan'] or stats['has_inf']:
                    f.write(f"  WARNING: Contains NaN or Inf values!\n")
        
        print(f"\nStatistics report saved to: {report_path}")
        
        return output_pnet, output_part, stats_dict_pnet


def load_training_files_from_yaml(samples_file):
    """Load and expand training file patterns from a YAML list."""
    with open(samples_file, 'r') as f:
        sample_entries = yaml.safe_load(f)
    
    if not sample_entries:
        raise ValueError(f"Samples file {samples_file} is empty")
    
    if not isinstance(sample_entries, list):
        raise ValueError(f"Samples file {samples_file} must contain a list of file paths or patterns")
    
    resolved_files = []
    samples_dir = os.path.dirname(os.path.abspath(samples_file))
    
    for entry in sample_entries:
        if isinstance(entry, str):
            pattern = entry
        elif isinstance(entry, dict):
            # Support simple dict entries like {path: "..."} or {file: "..."}
            pattern = entry.get('path') or entry.get('file')
            if pattern is None:
                continue
        else:
            continue
        
        pattern = os.path.expandvars(os.path.expanduser(pattern))
        if not os.path.isabs(pattern):
            pattern = os.path.abspath(os.path.join(samples_dir, pattern))
        
        matched = glob.glob(pattern)
        if matched:
            resolved_files.extend(matched)
        else:
            print(f"WARNING: Pattern {pattern} matched no files")
    
    resolved_files = sorted(set(resolved_files))
    
    if not resolved_files:
        raise ValueError(f"No files found using samples file {samples_file}")
    
    return resolved_files


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute standardization parameters from all training files'
    )
    parser.add_argument('--data-dir', type=str, 
                       default='/HEP/data/share/aleph/aleph-data/ntuples/mc',
                       help='Directory containing output_*_train.root files')
    parser.add_argument('--samples-file', type=str, default=None,
                       help='YAML file listing training files or glob patterns (overrides --data-dir)')
    parser.add_argument('--config-pnet', type=str,
                       default='configs/data_config_pnet.yaml',
                       help='Path to ParticleNet config template (relative to weaver dir)')
    parser.add_argument('--config-part', type=str,
                       default='configs/data_config_part.yaml',
                       help='Path to ParticleTransformer config template (relative to weaver dir)')
    parser.add_argument('--output-dir', type=str,
                       default='configs/standardized_configs',
                       help='Output directory for generated configs (relative to weaver dir)')
    parser.add_argument('--sample-fraction', type=float, default=0.1,
                       help='Fraction of events to use from each file (default: 0.1)')
    parser.add_argument('--max-events-per-file', type=int, default=50000,
                       help='Maximum events per file (default: 50000, set to 0 for no limit)')
    
    args = parser.parse_args()
    
    # Get weaver directory (where configs are and where we'll run from)
    weaver_dir = os.path.join(weavercoredir, 'weaver')
    
    # Handle config paths - if absolute, use as-is; otherwise relative to weaver_dir
    if os.path.isabs(args.config_pnet):
        config_pnet = args.config_pnet
    else:
        config_pnet = os.path.join(weaver_dir, args.config_pnet)
    
    if os.path.isabs(args.config_part):
        config_part = args.config_part
    else:
        config_part = os.path.join(weaver_dir, args.config_part)
    
    # Handle output dir - if absolute, use as-is; otherwise relative to weavercoredir
    # (since paths like "weaver/configs/..." are relative to weaver-core root)
    if os.path.isabs(args.output_dir):
        output_dir = os.path.abspath(args.output_dir)
    else:
        # If path starts with "weaver/", it's relative to weavercoredir, not weaver_dir
        if args.output_dir.startswith('weaver/'):
            output_dir = os.path.abspath(os.path.join(weavercoredir, args.output_dir))
        else:
            # Otherwise, assume it's relative to weaver_dir
            output_dir = os.path.abspath(os.path.join(weaver_dir, args.output_dir))
    
    # Resolve samples file if provided
    samples_file = None
    training_files = None
    data_source_desc = None
    
    if args.samples_file:
        if os.path.isabs(args.samples_file):
            samples_file = args.samples_file
        elif args.samples_file.startswith('weaver/'):
            samples_file = os.path.abspath(os.path.join(weavercoredir, args.samples_file))
        else:
            samples_file = os.path.abspath(os.path.join(weaver_dir, args.samples_file))
        
        if not os.path.exists(samples_file):
            raise FileNotFoundError(f"Samples file not found: {samples_file}")
        
        training_files = load_training_files_from_yaml(samples_file)
        data_source_desc = f"samples file {samples_file}"
    
    data_dir = args.data_dir
    if data_dir:
        data_dir = os.path.abspath(data_dir)
    if training_files is None:
        data_source_desc = data_dir
    
    # Change to weaver directory (where DataConfig expects to be)
    os.chdir(weaver_dir)
    
    # Set max_events_per_file
    max_events = args.max_events_per_file if args.max_events_per_file > 0 else None
    
    # Create computer
    computer = StandardizationComputer(
        data_dir=None if training_files is not None else data_dir,
        config_template_pnet=config_pnet,
        config_template_part=config_part,
        sample_fraction=args.sample_fraction,
        max_events_per_file=max_events,
        training_files=training_files,
        data_source_desc=data_source_desc
    )
    
    # Generate configs
    output_pnet, output_part, stats_dict = computer.generate_configs(output_dir)
    
    print(f"\n{'='*60}")
    print("Standardization Complete!")
    print(f"{'='*60}")
    print(f"\nGenerated configs:")
    print(f"  ParticleNet: {output_pnet}")
    print(f"  ParticleTransformer: {output_part}")
    print(f"\nNext steps:")
    print(f"  1. Review the generated configs")
    print(f"  2. Copy them to configs/ directory if satisfied")
    print(f"  3. Update run.py to use the new configs")


if __name__ == '__main__':
    main()

