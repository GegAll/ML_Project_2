import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from scipy.signal import find_peaks, butter, filtfilt, freqz, stft
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from experiment import post_process, extract_heartbeats, normalize_recordings, plot_heartbeats, calculate_statistics, grid_search, detect_irregularities_dbscan, plot_clusters_with_pca, plot_clusters_with_pca_3d, plot_irregular_heartbeats

def lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a high-pass Butterworth filter to the given data.

    Parameters:
    ----------
    data : numpy.ndarray
        The input signal.
    cutoff : float
        The cutoff frequency of the high-pass filter (in Hz).
    fs : float
        The sampling frequency of the signal (in Hz).
    order : int, optional
        The order of the Butterworth filter. Default is 5.

    Returns:
    -------
    numpy.ndarray
        The filtered signal.
    """
    # Design the high-pass filter
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def load_data(path):
    data = loadmat(path)
    values = data['val']

    return values

def apply_ica(values):

    """
    Applies ICA to the data

    Parameters:
    -----------
    path: string
        the path to the .mat file containing the recordings

    Returns:
    --------
    result: ndarray
        array of shape (4, recording length) with the signals after they were unmixed
        
    """

    # apply ICA on data
    ica = FastICA()
    result = np.transpose(ica.fit_transform(np.transpose(values)))

    # Make sure that the amplitudes ar4e positives for consistency (sometimes ICA makes them negative)
    results = []
    for arr in result:
        if abs(np.min(arr)) > abs(np.max(arr)):
            arr = -arr
        results.append(arr)
            
    return np.array(results)

def post_process(result, highcut, fs=1000):
    """
    Post process the data by using the bandpass filter

    Parameters:
    -----------
    result : ndarray
        the result of ICA
    lowcut : float
        The lower cutoff frequency of the bandpass filter in Hz.
    highcut : float
        The upper cutoff frequency of the bandpass filter in Hz.
    fs : int, optional
        Sampling frequency in Hz (default is 360).
    plot : bool, optional
        Whether to plot the filtered signal (default is False).

    Returns:
    --------
    results: ndarray
        An array of shape (4, recording length) with the signals after filtering.
    """
    results = []

    for i, component in enumerate(result):
        filtered_signal = lowpass_filter(component, highcut, fs)
        # filtered_signal = component
        results.append(filtered_signal)
        # print(len(find_peaks(filtered_signal)[0]))

    return np.array(results)

def plot_channels(sequence, name, fs=1000):
    X = np.linspace(0, len(sequence[0]) / fs, len(sequence[0]))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  

    for i, channel in enumerate(sequence):
        ax = axes[i // 2, i % 2]  # Row and column index for 2x2 grid
        ax.plot(X[:10000], channel[:10000], label=f"Channel {i+1} Filtered")
        ax.set_title(f"Filtered Signal (Channel: {i+1})")
        ax.legend()

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(f'figs/{name}')
    # plt.show()

if __name__ == '__main__':
    synthetic_path = 'data/real/'
    mothers = []
    fetuses = []
    min_length = 1000000000
    for i in range(25):
        print(f'{synthetic_path}a{str(i+1).zfill(2)}m.mat')
        file_path = f'{synthetic_path}a{str(i+1).zfill(2)}m.mat'
        results = load_data(file_path)
        results = post_process(results, 5, fs=1000)
        results = apply_ica(results)
        plot_channels(results, f'a{str(i+1).zfill(2)}m')
        # results = post_process(results, 100, fs=1000, plot=True)
        # print(results.shape)
        mother, fetus = extract_heartbeats(results, sampling_rate=1000)
        min_length = min(min_length, len(fetus))
        mothers.append(mother)
        fetuses.append(fetus)
        print()
    plot_heartbeats(mother_heartbeats=mothers, fetus_heartbeats=fetuses)
    
    fetal_heartbeat = normalize_recordings(fetuses, min_length)

    # Clustering extracted features
    fetal_heartbeat_statistics = []
    for heartbeat in fetuses:
        _, _, average_frequency, std_frequency, _, std_amplitude = calculate_statistics(heartbeat, 0, 1000,  print_statistics=False)
        fetal_heartbeat_statistics.append([average_frequency, std_frequency, std_amplitude])
    scalar = StandardScaler()
    scalar.fit(fetal_heartbeat_statistics)
    fetal_heartbeat_statistics = scalar.transform(fetal_heartbeat_statistics)
    eps, min_samples = grid_search(fetal_heartbeat_statistics)
    labels = detect_irregularities_dbscan(fetal_heartbeat_statistics, eps=eps, min_samples=min_samples)
    print(labels)
    plot_clusters_with_pca(fetal_heartbeat_statistics, labels, use_pca=True)
    plot_clusters_with_pca_3d(fetal_heartbeat_statistics, labels, use_pca=False)
    plot_irregular_heartbeats(fetal_heartbeat, labels)
