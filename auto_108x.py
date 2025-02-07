import argparse
import pandas as pd
import numpy as np
import os
import gc
from multiprocessing import Pool, cpu_count
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from torchsummary import summary

def get_args():
    parser = argparse.ArgumentParser(description="Train a ConvAutoencoder on genomic data")
    
    parser.add_argument('-b','--bed_file', type=str, default='filtered_genes.bed', help='Path to BED file')
    parser.add_argument('-wd','--directory_template', type=str, default='/group/sbs007/bdao/project/data/H3K4me3/wig/chr/{}', help='Directory template for data')
    parser.add_argument('-e','--extension', type=int, default=2000, help='Extension size for BED regions')
    parser.add_argument('-w','--window_size', type=int, default=300000, help='Window size for genomic data')
    parser.add_argument('-ed','--encoding_dim', type=int, default=1, help='Dimensionality of the encoded space')
    # In get_args(), change:
    parser.add_argument('-m','--model_save_path', type=str, 
                   default='./models_108x.pth',  # Changed from models/autoencoder.pth
                   help='Path to save the trained model')
    parser.add_argument('-mpc','--max_regions_per_chromosome', type=int, default=5, help='Max regions per chromosome')
    parser.add_argument('-t','--total_max_regions', type=int, default=100, help='Total max regions to process')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--reduce_lr_patience', type=int, default=10, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=50, help='Patience for early stopping')

    return parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preparation functions
def ref_ext(row):
    return [f'{row["chr"]}_{i}' for i in range(row['start'], row['end'] + 1)]

def lazy_load_data(directory):
    files = []
    for wig_file in os.listdir(directory):
        if wig_file.endswith('.wig'):
            files.append((os.path.join(directory, wig_file), wig_file))
    return files

def lazy_process_data(args):
    file_path, test, ref_set, label = args
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['chrom', 'start', 'end', 'score'], comment='#')
    matrix = np.zeros(len(ref_set), dtype=np.float32)

    for _, row in df.iterrows():
        for i in range(max(row['start'], test['start'].iloc[0]), min(row['end'], test['end'].iloc[0]) + 1):
            key = f'{row["chrom"]}_{i}'
            if key in ref_set:
                matrix[ref_set[key]] = row['score']

    return np.concatenate([matrix.reshape(1, -1), np.full((1, 1), label, dtype=np.int32)], axis=1)

def lazy_concatenate_dfs(df_list):
    df_list = [arr.astype(np.float32) for arr in df_list]
    return csr_matrix(np.vstack(df_list))

def data_preparation(directory, test, ref_list):
    start_time = time.time()
    ref_set = {v: i for i, v in enumerate(ref_list)}
    df_list = []

    for label in ['Healthy', 'CRC']:
        label_directory = os.path.join(directory, label)
        if not os.path.exists(label_directory):
            print(f"Warning: Directory {label_directory} does not exist. Skipping.")
            continue
        numeric_label = 0 if label == 'Healthy' else 1
        lazy_loaded_data = ((file_path, test, ref_set, numeric_label) for file_path, _ in lazy_load_data(label_directory))

        with Pool(cpu_count()) as pool:
            df_list.extend(pool.map(lazy_process_data, lazy_loaded_data))

    if not df_list:
        raise FileNotFoundError(f"No data found in directory: {directory}")

    matrix = lazy_concatenate_dfs(df_list)
    matrix.data[np.isnan(matrix.data)] = 0
    labels = matrix[:, -1].toarray().ravel().astype(np.int32)
    X = matrix[:, :-1].toarray().astype(np.float32)
    y = labels

    del df_list, matrix, labels
    gc.collect()

    end_time = time.time()
    print(f"Data preparation took {end_time - start_time:.2f} seconds.")
    return X, y

class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=1):
        super(ConvAutoencoder, self).__init__()
        
        # Initialize the encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=6, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, encoding_dim, kernel_size=5, stride=3)
        )
        
        # Initialize the decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(encoding_dim, 64, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, kernel_size=6, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 16, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 1, kernel_size=6, stride=3)
        )

    def forward(self, x):
        # Handle input dimensions
        if len(x.shape) == 4:
            x = x.squeeze(2)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Process through encoder
        encoded = self.encoder(x)
        
        # Process through decoder
        decoded = self.decoder(encoded)
        
        return decoded.squeeze(1)

    def get_encoding(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(2)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

def build_autoencoder(input_dim, encoding_dim, log_dir):
    # Create base model
    autoencoder = ConvAutoencoder(input_dim, encoding_dim)
    
    # Move to device
    autoencoder = autoencoder.to(device)
    
    # Try to print model summary
    try:
        print("Model Summary:")
        summary(autoencoder, input_size=(1, input_dim))
    except Exception as e:
        print(f"Warning: Could not generate model summary: {e}")
    
    # Wrap with DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        autoencoder = nn.DataParallel(autoencoder)

    # Create optimizer and criterion
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir=log_dir)

    return autoencoder, optimizer, criterion, writer

def train_autoencoder(autoencoder, optimizer, criterion, X_train, X_val, epochs, batch_size, log_dir, model_save_path, reduce_lr_patience=10, early_stopping_patience=50):
    autoencoder.train()
    writer = SummaryWriter(log_dir=log_dir)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=reduce_lr_patience, factor=0.1, verbose=True)

    # Ensure data is on correct device
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train)
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val)
    
    X_train = X_train.to(device)
    X_val = X_val.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train[indices]

            optimizer.zero_grad()
            
            # Forward pass
            outputs = autoencoder(batch_x)
            
            # Handle size mismatch if necessary
            if outputs.size() != batch_x.size():
                if len(outputs.shape) == 3:  # 3D tensor
                    min_size = min(outputs.size(2), batch_x.size(2))
                    outputs = outputs[:, :, :min_size]
                    batch_x = batch_x[:, :, :min_size]
                elif len(outputs.shape) == 2:  # 2D tensor
                    min_size = min(outputs.size(1), batch_x.size(1))
                    outputs = outputs[:, :min_size]
                    batch_x = batch_x[:, :min_size]

            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation phase
        with torch.no_grad():
            val_outputs = autoencoder(X_val)
            
            # Handle validation size mismatch
            if val_outputs.size() != X_val.size():
                if len(val_outputs.shape) == 3:
                    min_size = min(val_outputs.size(2), X_val.size(2))
                    val_outputs = val_outputs[:, :, :min_size]
                    X_val_temp = X_val[:, :, :min_size]
                elif len(val_outputs.shape) == 2:
                    min_size = min(val_outputs.size(1), X_val.size(1))
                    val_outputs = val_outputs[:, :min_size]
                    X_val_temp = X_val[:, :min_size]
            else:
                X_val_temp = X_val

            val_loss = criterion(val_outputs, X_val_temp).item()

        scheduler.step(val_loss)
        
        # Logging
        writer.add_scalar('Loss/train', epoch_loss / len(X_train), epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(X_train):.4f}, Val Loss: {val_loss:.4f}')

        # Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(autoencoder.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print(f'Early stopping on epoch {epoch + 1}')
                break

    writer.close()
    # Load best model
    autoencoder.load_state_dict(torch.load(model_save_path))

def extend_bed_regions(bed_file, extension=2000):
    regions = pd.read_csv(bed_file, sep='\t', header=None, names=['chr', 'start', 'end', 'name', 'gene', 'strand'])
    regions['start'] = regions['start'] - extension
    regions['end'] = regions['end'] + extension
    regions.loc[regions['start'] < 0, 'start'] = 0  # Ensure start is not negative
    return regions

def define_windows(regions, window_size=300000):
    windows = []
    current_window = None

    for _, region in regions.iterrows():
        chr_name = region['chr']
        region_start = region['start']

        # Check if a new window should be started
        if current_window is None or region_start > current_window['end'] or chr_name != current_window['chr']:
            # Start a new window if:
            # 1. It's the first window.
            # 2. The gene start is beyond the current window end.
            # 3. The gene is on a different chromosome than the current window.
            current_window = {'chr': chr_name, 'start': region_start, 'end': region_start + window_size - 1}
            windows.append(current_window)

    return pd.DataFrame(windows)

def load_pretrained_model(model_path, input_dim, encoding_dim):
    if not os.path.exists(model_path):
        print(f"No pretrained model found at {model_path}")
        return None
        
    try:
        model = ConvAutoencoder(input_dim, encoding_dim)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None

def main():
    args = get_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    os.makedirs('./auto_logs', exist_ok=True)

    # Unpack arguments for easy use
    bed_file = args.bed_file
    directory_template = args.directory_template
    extension = args.extension
    window_size = args.window_size
    encoding_dim = args.encoding_dim
    model_save_path = args.model_save_path
    max_regions_per_chromosome = args.max_regions_per_chromosome
    total_max_regions = args.total_max_regions
    epochs = args.epochs
    batch_size = args.batch_size
    reduce_lr_patience = args.reduce_lr_patience
    early_stopping_patience = args.early_stopping_patience

    region_count = 0

    # Read and extend BED file regions
    regions = extend_bed_regions(bed_file, extension)

    # Define windows based on the overlapping genes' ranges
    windows = define_windows(regions, window_size)

    last_chr = None
    regions_this_chromosome = 0

    for index, window in windows.iterrows():
        chr_name = window['chr']

        # Reset region count per chromosome when switching chromosomes
        if last_chr != chr_name:
            last_chr = chr_name
            regions_this_chromosome = 0

        # Check if maximum regions per chromosome or total has been reached
        if regions_this_chromosome >= max_regions_per_chromosome or region_count >= total_max_regions:
            continue

        start = window['start']
        end = window['end']
        test = pd.DataFrame({'chr': [chr_name], 'start': [start], 'end': [end]})
        test['start'] = test['start'].astype(int)
        test['end'] = test['end'].astype(int)
        ref = test.apply(ref_ext, axis=1)
        ref_list = [item for sublist in ref for item in sublist]

        directory = directory_template.format(chr_name)

        X, y = data_preparation(directory, test, ref_list)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, X_val = torch.tensor(X_train).to(device), torch.tensor(X_val).to(device)  # Move tensors to device
        y_train, y_val = torch.tensor(y_train).to(device), torch.tensor(y_val).to(device)  # Move tensors to device

        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        # In main(), change:
        log_dir_autoencoder = f"./auto_logs/{timestamp}_{chr_name}_{start}-{end}/"  # Changed from autoencoder_logs/

        autoencoder, optimizer, criterion, writer = build_autoencoder(X_train.shape[1], encoding_dim, log_dir_autoencoder)

        # Load the previously trained model if it exists
        if os.path.exists(model_save_path):
            autoencoder.load_state_dict(torch.load(model_save_path))
        else:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        train_autoencoder(autoencoder, optimizer, criterion, X_train, X_val, epochs=epochs, batch_size=batch_size, log_dir=log_dir_autoencoder, model_save_path=model_save_path, reduce_lr_patience=reduce_lr_patience, early_stopping_patience=early_stopping_patience)

        regions_this_chromosome += 1
        region_count += 1

        if region_count >= total_max_regions:
            print(f"Reached the limit of {total_max_regions} regions. Stopping training.")
            break

    print(f"Training completed with {region_count} regions processed.")

if __name__ == "__main__":
    main()

print('This is cnn')
