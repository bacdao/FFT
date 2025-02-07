import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from multiprocessing import Pool, cpu_count
import time
import pandas as pd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from auto_108x import ConvAutoencoder, ref_ext, lazy_process_data

def set_publication_style():
    plt.style.use('default')  # Reset to default style
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 18,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.figsize': (12, 10),
        'axes.grid': False,  # Removed grid
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

def plot_chipseq_and_fft(original_data, gene_symbol, save_path=None):
    set_publication_style()
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Plot original signal
    ax1 = fig.add_subplot(gs[0])
    x_axis = np.arange(len(original_data))
    ax1.plot(x_axis, original_data, 
            color='#2166AC',
            linewidth=1.5, 
            alpha=0.8,
            label='H3K4me3 Signal')
    
    # Add 'A' label
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, 
             fontsize=20, fontweight='bold')
    
    ax1.set_xlabel('Genomic Position (bp)')
    ax1.set_ylabel('Signal Intensity')
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax1.grid(False)  # Removed grid
    ax1.legend(frameon=True, fancybox=True, shadow=False, loc='upper right')
    
    # Calculate FFT components
    fft_result = np.fft.rfft(original_data)
    frequencies = np.fft.rfftfreq(len(original_data))
    magnitudes = np.abs(fft_result)
    
    # Find top N magnitude components
    N = len(original_data) // 216  # Number of top components to highlight
    top_indices = np.argsort(magnitudes)[-N:]
    
    # Create mask for background (non-top) components
    background_mask = np.ones(len(frequencies), dtype=bool)
    background_mask[top_indices] = False
    
    # Plot FFT magnitude spectrum
    ax2 = fig.add_subplot(gs[1])
    
    # Plot background components in lighter color
    ax2.semilogy(frequencies[background_mask], magnitudes[background_mask],
                 color='#B2182B',
                 linewidth=1.0,
                 alpha=0.4,
                 label='FFT Transformation')
    
    # Plot top components in green
    ax2.semilogy(frequencies[top_indices], magnitudes[top_indices],
                 color='#2ca02c',  # Kept green color
                 linewidth=2.0,
                 alpha=1.0,
                 label=f'Top highest magnitude components')
    
    # Create inset for zoomed view
    axins = inset_axes(ax2, width="40%", height="40%",
                      bbox_to_anchor=(0.5, 0.5, 0.95, 0.95),
                      bbox_transform=ax2.transAxes)
    
    # Focus on the first 2.5% of frequencies for zoom
    zoom_end_idx = int(len(frequencies) * 0.025)
    zoom_freqs = frequencies[:zoom_end_idx]
    zoom_mags = magnitudes[:zoom_end_idx]
    
    # Create zoom mask for top components
    zoom_top_mask = np.isin(frequencies[:zoom_end_idx], frequencies[top_indices])
    zoom_background_mask = ~zoom_top_mask
    
    # Plot zoomed view
    axins.semilogy(zoom_freqs[zoom_background_mask], zoom_mags[zoom_background_mask],
                   color='#B2182B',
                   linewidth=1.0,
                   alpha=0.4)
    
    axins.semilogy(zoom_freqs[zoom_top_mask], zoom_mags[zoom_top_mask],
                   color='#2ca02c',  # Changed to green to match main plot
                   linewidth=2.0,
                   alpha=1.0)
    
    # Customize zoom region
    axins.grid(False)  # Removed grid
    axins.set_xlabel('Frequency', fontsize=10)
    axins.set_title('Zoom', fontsize=10, pad=8)
    
    # Format ticks for better readability
    def format_tick(x, p):
        return f'{x:.3f}'
    
    axins.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick))
    axins.xaxis.set_major_locator(ticker.MaxNLocator(4))
    axins.tick_params(axis='x', rotation=45)
    
    # Add connecting lines with custom style
    mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", 
               ec="#666666", ls="--", alpha=0.6)
    
    # Add 'B' label
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, 
             fontsize=20, fontweight='bold')
    
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude (log scale)')
    ax2.set_xlim(0, 0.1)
    ax2.grid(False)  # Removed grid
    ax2.legend(frameon=True, fancybox=True, shadow=False,
              loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the figure
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def lazy_load_data(directory):
    return [(os.path.join(directory, f), f) for f in os.listdir(directory) if f.endswith('.wig')]

def prepare_sample_data_multiprocessing(directory, test, ref_list):
    ref_set = {v: i for i, v in enumerate(ref_list)}
    lazy_loaded_data = lazy_load_data(directory)
    
    if not lazy_loaded_data:
        raise FileNotFoundError(f"No .wig files found in directory: {directory}")

    with Pool(cpu_count()) as pool:
        data_list = pool.map(lazy_process_data, ((file_path, test, ref_set, 1) for file_path, _ in lazy_loaded_data))

    sample_data = np.vstack(data_list)
    return sample_data[:, :-1].astype(np.float64)

def main(gene_symbol):
    bed_file = 'gencode.v46lift37.annotation.bed'
    bed_df = pd.read_csv(bed_file, sep='\t', names=['chr', 'start', 'end', 'id', 'gene_symbol', 'strand'])
    
    gene_info = bed_df[bed_df['gene_symbol'] == gene_symbol].iloc[0]
    chr_name = gene_info['chr']
    start = max(gene_info['start'] - 2000, 0)
    end = start + 300000
    
    directory = f'/group/sbs007/bdao/project/data/H3K4me3/wig/independence/chr/{chr_name}/CRC'
    test_data = pd.DataFrame({'chr': [chr_name], 'start': [start], 'end': [end]})
    ref_list = test_data.apply(ref_ext, axis=1).explode().tolist()
    
    try:
        X = prepare_sample_data_multiprocessing(directory, test_data, ref_list)
        original_sample = X[0]
        
        # Create output directory if it doesn't exist
        output_dir = 'publish_test/figure1'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plots
        plot_chipseq_and_fft(
            original_sample, 
            gene_symbol,
            save_path=os.path.join(output_dir, f'chipseq_signal_fft_{gene_symbol}.png')
        )
        
    except Exception as e:
        print(f"Error processing {gene_symbol}: {str(e)}")
        raise

if __name__ == "__main__":
    main('GAPDH')