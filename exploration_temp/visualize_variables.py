#!/usr/bin/env python3
"""
Generate visualizations for all variables in the dataset.
Creates distribution plots, statistics summaries, and standardization comparisons.
"""

import os
import sys
import glob
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import yaml

# Add weaver-core to path
thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../..'))
sys.path.append(os.path.join(weavercoredir, 'weaver'))
sys.path.append(weavercoredir)

from weaver.utils.data.config import DataConfig
from weaver.utils.data.fileio import _read_files
from weaver.utils.data.preprocess import _apply_selection, _build_new_variables
from weaver.utils.data.tools import _get_variable_names

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class VariableVisualizer:
    """Generate visualizations for variables."""
    
    def __init__(self, data_dir, config_path, output_dir, 
                 sample_fraction=0.05, max_events_per_file=20000):
        """
        Args:
            data_dir: Directory containing output_*_train.root files
            config_path: Path to data config file
            output_dir: Directory to save plots
            sample_fraction: Fraction of events to use (default: 0.05 = 5%)
            max_events_per_file: Maximum events per file
        """
        self.data_dir = data_dir
        self.config_path = config_path
        self.output_dir = output_dir
        self.sample_fraction = sample_fraction
        self.max_events_per_file = max_events_per_file
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'distributions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'standardized'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'summary'), exist_ok=True)
        
        # Find training files
        pattern = os.path.join(data_dir, 'output_*_train.root')
        self.training_files = sorted(glob.glob(pattern))[:10]  # Use first 10 files for speed
        
        print(f"Found {len(self.training_files)} training files (using first 10 for visualization)")
    
    def load_data_sample(self, data_config):
        """Load a sample of data from files."""
        print(f"\nLoading data from {len(self.training_files)} files...")
        
        # Load branches needed
        all_input_vars = set()
        for var_list in data_config.input_dicts.values():
            all_input_vars.update(var_list)
        
        load_branches = all_input_vars.copy()
        if data_config.selection:
            load_branches.update(_get_variable_names(data_config.selection))
        
        for var_name, expr in data_config.var_funcs.items():
            load_branches.update(_get_variable_names(expr))
        
        # Load data from files
        all_tables = []
        for filepath in tqdm(self.training_files, desc="Loading files"):
            try:
                import uproot
                with uproot.open(filepath) as f:
                    treename = self._get_treename(f)
                    tree = f[treename]
                    total_events = tree.num_entries
                    num_events = min(int(total_events * self.sample_fraction), 
                                   self.max_events_per_file or total_events)
                    load_range = (0, num_events / total_events) if total_events > 0 else (0, 1)
                
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
            except Exception as e:
                print(f"  WARNING: Failed to load {filepath}: {e}")
                continue
        
        if len(all_tables) == 0:
            raise ValueError("No data loaded from any files!")
        
        # Concatenate
        from weaver.utils.data.tools import _concat
        combined_table = _concat(all_tables)
        print(f"Total events loaded: {len(combined_table)}")
        
        # Apply preprocessing
        if data_config.selection:
            combined_table = _apply_selection(combined_table, data_config.selection, 
                                           funcs=data_config.var_funcs)
        combined_table = _build_new_variables(combined_table, data_config.var_funcs)
        
        print(f"Events after selection: {len(combined_table)}")
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
    
    def get_variable_data(self, table, var_name):
        """Extract and flatten variable data."""
        if var_name not in table.fields:
            return None
        
        var_data = table[var_name]
        
        # Flatten if jagged array
        if isinstance(var_data, ak.Array):
            flat_data = ak.flatten(var_data, axis=None)
            flat_data = ak.to_numpy(flat_data)
        else:
            flat_data = np.array(var_data)
        
        # Remove NaN and Inf
        flat_data = flat_data[np.isfinite(flat_data)]
        
        return flat_data if len(flat_data) > 0 else None
    
    def plot_distribution(self, data, var_name, stats_dict=None, save_path=None):
        """Plot distribution of a variable."""
        if data is None or len(data) == 0:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Remove outliers for better visualization (keep 99% of data)
        if len(data) > 1000:
            q01, q99 = np.percentile(data, [1, 99])
            data_clean = data[(data >= q01) & (data <= q99)]
        else:
            data_clean = data
        
        # Plot 1: Histogram
        ax1 = axes[0]
        n, bins, patches = ax1.hist(data_clean, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel(var_name, fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'Distribution: {var_name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        if stats_dict and var_name in stats_dict:
            stats = stats_dict[var_name]
            stats_text = f"Mean: {stats['mean']:.4f}\n"
            stats_text += f"Median: {stats['median']:.4f}\n"
            stats_text += f"Std: {stats['std']:.4f}\n"
            stats_text += f"Min: {stats['min']:.4f}\n"
            stats_text += f"Max: {stats['max']:.4f}"
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Log scale (if appropriate)
        ax2 = axes[1]
        if np.min(data_clean) > 0:
            ax2.hist(data_clean, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.set_yscale('log')
            ax2.set_xlabel(var_name, fontsize=12)
            ax2.set_ylabel('Count (log scale)', fontsize=12)
            ax2.set_title(f'Distribution (log scale): {var_name}', fontsize=14, fontweight='bold')
        else:
            ax2.hist(data_clean, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.set_xlabel(var_name, fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title(f'Distribution (zoomed): {var_name}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_standardized_comparison(self, data, var_name, stats_dict, save_path=None):
        """Plot original vs standardized distribution."""
        if data is None or len(data) == 0 or var_name not in stats_dict:
            return
        
        stats = stats_dict[var_name]
        center = stats['center']
        scale = stats['scale']
        
        # Standardize
        standardized = (data - center) * scale
        # Clip to [-5, 5] as in weaver
        standardized = np.clip(standardized, -5, 5)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Remove outliers for visualization
        if len(data) > 1000:
            q01, q99 = np.percentile(data, [1, 99])
            data_clean = data[(data >= q01) & (data <= q99)]
        else:
            data_clean = data
        
        # Original distribution
        ax1 = axes[0]
        ax1.hist(data_clean, bins=100, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        ax1.axvline(center, color='red', linestyle='--', linewidth=2, label=f'Center (median): {center:.4f}')
        ax1.set_xlabel(var_name, fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'Original Distribution: {var_name}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Standardized distribution
        ax2 = axes[1]
        ax2.hist(standardized, bins=100, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Center (0)')
        ax2.axvline(-5, color='orange', linestyle='--', linewidth=1, label='Clip bounds')
        ax2.axvline(5, color='orange', linestyle='--', linewidth=1)
        ax2.set_xlabel(f'{var_name} (standardized)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'Standardized Distribution: {var_name}\n(center={center:.4f}, scale={scale:.4f})', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_statistics_summary(self, stats_dict, save_path=None):
        """Plot summary statistics for all variables."""
        if not stats_dict:
            return
        
        # Prepare data
        var_names = list(stats_dict.keys())
        centers = [stats_dict[v]['center'] for v in var_names]
        scales = [stats_dict[v]['scale'] for v in var_names]
        means = [stats_dict[v]['mean'] for v in var_names]
        stds = [stats_dict[v]['std'] for v in var_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Centers
        ax1 = axes[0, 0]
        ax1.barh(range(len(var_names)), centers, alpha=0.7)
        ax1.set_yticks(range(len(var_names)))
        ax1.set_yticklabels(var_names, fontsize=8)
        ax1.set_xlabel('Center (median)', fontsize=12)
        ax1.set_title('Standardization Centers', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(0, color='red', linestyle='--', linewidth=1)
        
        # Plot 2: Scales
        ax2 = axes[0, 1]
        ax2.barh(range(len(var_names)), scales, alpha=0.7, color='green')
        ax2.set_yticks(range(len(var_names)))
        ax2.set_yticklabels(var_names, fontsize=8)
        ax2.set_xlabel('Scale', fontsize=12)
        ax2.set_title('Standardization Scales', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Means
        ax3 = axes[1, 0]
        ax3.barh(range(len(var_names)), means, alpha=0.7, color='orange')
        ax3.set_yticks(range(len(var_names)))
        ax3.set_yticklabels(var_names, fontsize=8)
        ax3.set_xlabel('Mean', fontsize=12)
        ax3.set_title('Variable Means', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Standard deviations
        ax4 = axes[1, 1]
        ax4.barh(range(len(var_names)), stds, alpha=0.7, color='purple')
        ax4.set_yticks(range(len(var_names)))
        ax4.set_yticklabels(var_names, fontsize=8)
        ax4.set_xlabel('Standard Deviation', fontsize=12)
        ax4.set_title('Variable Standard Deviations', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def load_statistics_from_config(self, config_path):
        """Load statistics from standardized YAML config."""
        stats_dict = {}
        
        if not os.path.exists(config_path):
            # Try to find it in standardized_configs directory
            alt_path = os.path.join(os.path.dirname(self.config_path), 
                                   'standardized_configs', 'data_config_pnet_standardized.yaml')
            if os.path.exists(alt_path):
                config_path = alt_path
            else:
                print(f"Warning: Config file not found: {config_path}")
                return stats_dict
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract standardization parameters from config
            for input_group in config.get('inputs', {}).values():
                for var_entry in input_group.get('vars', []):
                    if isinstance(var_entry, list) and len(var_entry) >= 3:
                        var_name = var_entry[0]
                        center = var_entry[1]
                        scale = var_entry[2]
                        stats_dict[var_name] = {
                            'center': center,
                            'scale': scale,
                            'median': center,
                        }
        except Exception as e:
            print(f"Error loading config: {e}")
        
        return stats_dict
    
    def generate_all_plots(self):
        """Generate all visualizations."""
        print(f"\n{'='*60}")
        print("Generating Variable Visualizations")
        print(f"{'='*60}")
        
        # Load config
        data_config = DataConfig.load(self.config_path)
        
        # Load statistics from standardized config
        config_path = os.path.join(os.path.dirname(self.config_path), 
                                   'standardized_configs', 'data_config_pnet_standardized.yaml')
        stats_dict = self.load_statistics_from_config(config_path)
        print(f"Loaded statistics for {len(stats_dict)} variables")
        
        # Load data
        print("\nLoading data sample...")
        table = self.load_data_sample(data_config)
        
        # Get all input variables
        all_input_vars = set()
        for var_list in data_config.input_dicts.values():
            all_input_vars.update(var_list)
        
        print(f"\nGenerating plots for {len(all_input_vars)} variables...")
        
        # Generate distribution plots
        for var_name in tqdm(sorted(all_input_vars), desc="Distribution plots"):
            data = self.get_variable_data(table, var_name)
            if data is not None:
                save_path = os.path.join(self.output_dir, 'distributions', f'{var_name}.png')
                self.plot_distribution(data, var_name, stats_dict, save_path)
        
        # Generate standardized comparison plots
        for var_name in tqdm(sorted(all_input_vars), desc="Standardized plots"):
            data = self.get_variable_data(table, var_name)
            if data is not None and var_name in stats_dict:
                save_path = os.path.join(self.output_dir, 'standardized', f'{var_name}_standardized.png')
                self.plot_standardized_comparison(data, var_name, stats_dict, save_path)
        
        # Generate summary plot
        if stats_dict:
            summary_path = os.path.join(self.output_dir, 'summary', 'statistics_summary.png')
            self.plot_statistics_summary(stats_dict, summary_path)
            print(f"\nSummary plot saved to: {summary_path}")
        
        print(f"\n{'='*60}")
        print("Visualization Complete!")
        print(f"{'='*60}")
        print(f"\nPlots saved to:")
        print(f"  Distributions: {os.path.join(self.output_dir, 'distributions')}")
        print(f"  Standardized: {os.path.join(self.output_dir, 'standardized')}")
        print(f"  Summary: {os.path.join(self.output_dir, 'summary')}")


def main():
    """Main function."""
    import argparse
    
    # Get weavercoredir (needed for paths)
    thisdir = os.path.abspath(os.path.dirname(__file__))
    weavercoredir = os.path.abspath(os.path.join(thisdir, '../..'))
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations for all variables'
    )
    parser.add_argument('--data-dir', type=str,
                       default='/HEP/data/share/aleph/aleph-data/ntuples/mc',
                       help='Directory containing output_*_train.root files')
    parser.add_argument('--config', type=str,
                       default='configs/data_config_pnet.yaml',
                       help='Path to data config file (relative to weaver dir)')
    parser.add_argument('--output-dir', type=str,
                       default='exploration_temp/plots',
                       help='Output directory for plots')
    parser.add_argument('--sample-fraction', type=float, default=0.05,
                       help='Fraction of events to use (default: 0.05)')
    parser.add_argument('--max-events-per-file', type=int, default=20000,
                       help='Maximum events per file (default: 20000)')
    
    args = parser.parse_args()
    
    # Get weaver directory (where configs are)
    weaver_dir = os.path.join(weavercoredir, 'weaver')
    
    # Handle paths
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        config_path = os.path.join(weaver_dir, args.config)
    
    if os.path.isabs(args.output_dir):
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(weavercoredir, args.output_dir)
    
    # Change to weaver directory
    os.chdir(weaver_dir)
    
    # Create visualizer
    visualizer = VariableVisualizer(
        data_dir=args.data_dir,
        config_path=config_path,
        output_dir=output_dir,
        sample_fraction=args.sample_fraction,
        max_events_per_file=args.max_events_per_file
    )
    
    # Generate plots
    visualizer.generate_all_plots()


if __name__ == '__main__':
    main()

