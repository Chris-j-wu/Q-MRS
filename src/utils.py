import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

def spec_fft(x, ax=0):
    '''
    takes in FID and performs Fourier transformation
    '''
    return np.fft.fftshift(np.fft.fft(x, axis=ax), axes=ax)

def fid_ifft(x, ax=0):
    '''
    takes in FID and performs inverse Fourier transformation
    '''
    return np.fft.ifft(np.fft.ifftshift(x, axes=ax), axis=ax)

def find_nearest(array, value):
    '''
    takes in an array and a number, finds the index of of the element in the array closest to number
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def zero_fill(fid_array, factor):
    """
    zero-fill a 2D numpy array of complex-valued Free Induction Decay (FID) signals along the first dimension,
    with each column representing a different FID signal. The zero-filling is applied by an arbitrary factor.

    Parameters:
    - fid_array (np.ndarray): 2D numpy array containing complex-valued FID signals,
                              where each column represents an FID signal.
    - factor (int): Factor by which to zero-fill each FID signal along the first dimension.

    Returns:
    - zero_filled_array (np.ndarray): 2D numpy array containing zero-filled complex-valued FID signals.
    """
    original_length = fid_array.shape[0]
    num_fids = fid_array.shape[1]
    new_length = original_length * factor
    
    # Ensure the new array is complex and matches the shape adjustment
    zero_filled_array = np.zeros((int(new_length), num_fids),  dtype=np.complex64)
    zero_filled_array[:original_length, :] = fid_array
    return zero_filled_array

def prepare_data(data, PPMST=4.2, PPMEND=0.5):
    '''
    prepare data for input into the neural network
    Parameters:
    - PPMST (int): Starting ppm for cropping the spectrum.
    - PPMEND (int): Ending ppm.
    '''
    idx_start = find_nearest(data.ppm_array, PPMST)
    idx_end = find_nearest(data.ppm_array, PPMEND)
    idx_start_zf = find_nearest(data.ppm_array_zf, PPMST)
    idx_end_zf = find_nearest(data.ppm_array_zf, PPMEND)

    # Truncate spectra and take real component
    specs_OFF = data.specs_OFF[:, idx_end:idx_start].real
    specs_DIFF = data.specs_DIFF[:, idx_end:idx_start].real

    specs_zf_OFF = data.specs_zf_OFF[:, idx_end_zf:idx_start_zf].real
    specs_zf_DIFF = data.specs_zf_DIFF[:, idx_end_zf:idx_start_zf].real

    # Normalize spectra
    specs_OFF = specs_OFF / np.max(np.abs(specs_OFF), axis=1)[:, np.newaxis]
    specs_DIFF = specs_DIFF / np.max(np.abs(specs_DIFF), axis=1)[:, np.newaxis]

    specs_OFF_zf = specs_zf_OFF / np.max(np.abs(specs_zf_OFF), axis=1)[:, np.newaxis]
    specs_DIFF_zf = specs_zf_DIFF / np.max(np.abs(specs_zf_DIFF), axis=1)[:, np.newaxis]

    X_train_vivo_input = np.stack((specs_OFF, specs_DIFF), -1)
    X_train_vivo = np.stack((specs_OFF_zf, specs_DIFF_zf), -1)

    return X_train_vivo_input, X_train_vivo

def freeze_layers(model, trainable_layer_names):
    model.trainable = True
    for layer in model.layers:
        if layer.name in trainable_layer_names:
            continue
        layer.trainable = False

def add_spectral_loss(model, PPMGAP=None, PPMEND=0.5, ppm_array=None):
    '''
    - PPMGAP (tuple with two elements): Specfies part of the spectrum to be masked/ignored during optimization. Similar to the gap parameters in LCModel
    '''
    input_specs = model.inputs[1]
    output_specs = model.output[1]

    if PPMGAP and ppm_array is not None:
        gap_idx_end = find_nearest(ppm_array, PPMGAP[0]) - find_nearest(ppm_array, PPMEND)
        gap_idx_start = find_nearest(ppm_array, PPMGAP[1]) - find_nearest(ppm_array, PPMEND)
    
        mask = tf.ones_like(input_specs[:, :, 0])
        mask_slice = tf.zeros_like(mask[:, gap_idx_start:gap_idx_end])
        mask = tf.concat([
            mask[:, :gap_idx_start],
            mask_slice,
            mask[:, gap_idx_end:]
        ], axis=1)

        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.tile(mask, [1, 1, 2])

        mask_channel = tf.expand_dims(mask[:, :, 1], axis=-1)  # mask only the DIFF channel

        input_specs = tf.concat([
            input_specs[:, :, :1],                      # keep OFF channel
            input_specs[:, :, 1:] * mask_channel        # mask DIFF channel
        ], axis=-1)

        output_specs = tf.concat([
            output_specs[:, :, :1],
            output_specs[:, :, 1:] * mask_channel
        ], axis=-1)

    loss_spectral = K.mean(
        K.mean(K.square(input_specs - output_specs), axis=-1),
        axis=-1
    )

    model.add_loss(loss_spectral)

def compile_model(model, lr=1e-4, opt='adam'):
    if opt == 'adamW':
        optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay = 0.005, amsgrad=False)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=False)
    model.compile(optimizer=optimizer, loss=None)

def plot_spectra(target_spectra, ppm_array, pred_specs=None, pred_params=None, low_vals=None, high_vals=None, save_path=None):

    def decimal_to_int(x, pos):
        return f'{int(x)}' if x.is_integer() else f'{x}'

    METABOLITE_LIST = ['Cr', 'PCr', 'GABA', 'Glu', 'Gln', 'GSH', 'Lac', 'NAA', 'NAAG', 'PCh', 'GPC', 'PE', 'Asc', 'Asp', \
                                'mI', 'sI', 'Tau', '-CrCH2', 'MM09', 'MM12', 'MM14', 'MM17', 'MM20', 'Lip09', 'Lip13', 'Lip20', 'MM3co']
    
    if pred_params is not None:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])       
    else:
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 1)

    # Subplots for OFF and DIFF spectra
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    target_OFF = target_spectra[:, 0]
    target_DIFF = target_spectra[:, 1]    
    pred_OFF = pred_specs[:, 0] if pred_specs is not None else None
    pred_DIFF = pred_specs[:, 1] if pred_specs is not None else None

    # OFF subplot
    ax0.plot(ppm_array, target_OFF, label='Target', linewidth=1.75)
    if pred_OFF is not None:
        ax0.plot(ppm_array, pred_OFF, linestyle='--', label='Fit', linewidth=1.75)

    ax0.set_xlim(0.5, 4.0)
    ax0.xaxis.set_major_formatter(FuncFormatter(decimal_to_int))
    ax0.set_ylim(-0.2, 1.0)
    ax0.legend()
    ax0.grid()
    ax0.invert_xaxis()
    ax0.set_xlabel('ppm')
    ax0.set_title('edit-OFF', fontweight='bold')
    
    # DIFF subplot
    ax1.plot(ppm_array, target_DIFF, label='Target', linewidth=1.75)
    if pred_DIFF is not None:
        ax1.plot(ppm_array, pred_DIFF, linestyle='--', label='Fit', linewidth=1.75)

    ax1.set_xlim(0.5, 4.0)
    ax1.xaxis.set_major_formatter(FuncFormatter(decimal_to_int))
    ax1.set_ylim(-0.8, 0.4)
    ax1.grid()
    ax1.invert_xaxis()
    ax1.set_xlabel('ppm')
    ax1.set_title('DIFF', fontweight='bold')

    # Bar graph for predicted metabolite amplitudes
    if pred_params is not None:
        ax2 = fig.add_subplot(gs[:, 1])
    
        pred_params = pred_params * (high_vals - low_vals) + low_vals # unscale parameters
        pred_amps = pred_params[:len(METABOLITE_LIST)]
        pred_amps = pred_amps / (pred_amps[0] + pred_amps[1]) # reference to tCr

        ax2.barh(METABOLITE_LIST, pred_amps, color='rebeccapurple')
        ax2.set_xlabel('1/[tCr]')
        ax2.set_title('Estimated Conc.', fontweight='bold')

        ax2.set_axisbelow(True)
        ax2.grid(axis='x')
        ax2.invert_yaxis()

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()