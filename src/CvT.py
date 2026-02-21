import math
from utils import find_nearest
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class ConvAttention(Layer):
    def __init__(self, dim, heads, dim_head, kernel_size=3,
                 q_stride=1, k_stride=2, v_stride=2, dropout=0.0,
                 last_stage=False):
        super().__init__()
        self.last_stage = last_stage
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        # Convolutional projection
        self.to_q = SeparableConv1D(self.inner_dim, kernel_size=kernel_size, strides=q_stride, padding='same')
        self.to_k = SeparableConv1D(self.inner_dim, kernel_size=kernel_size, strides=k_stride, padding='same')
        self.to_v = SeparableConv1D(self.inner_dim, kernel_size=kernel_size, strides=v_stride, padding='same')

        self.to_out = tf.keras.Sequential([
            Dense(dim),
        ]) if (heads > 1 or dim_head != dim) else Layer()

    def call(self, x):
        # x: (B, L, D) 
        b = tf.shape(x)[0]
        h = self.heads
        d = self.dim_head

        # Apply convolutions to get Q, K, V
        q = self.to_q(x)  # (B, L, H*D) batch, seq_len, num heads, dimension
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape to (B, H, L, D)
        q = tf.reshape(q, (b, -1, h, d))
        k = tf.reshape(k, (b, -1, h, d))
        v = tf.reshape(v, (b, -1, h, d))

        q = tf.transpose(q, perm=[0, 2, 1, 3])  # (B, H, L, D)
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (b, -1, h * d))

        return self.to_out(out)
    
def CvTBlock(x, embed_dim, num_heads, ff_dim):
    x_norm = LayerNormalization()(x)
    attn_output = ConvAttention(
        dim=embed_dim,
        heads=num_heads,
        dim_head=embed_dim // num_heads,
        kernel_size=3,
        q_stride=1, k_stride=2, v_stride=2,
        dropout=0.0
    )(x_norm)
    x = x + attn_output

    x_norm = LayerNormalization()(x)
    ffn = Dense(ff_dim, activation='gelu')(x_norm)
    ffn = Dense(embed_dim)(ffn)
    x = x + ffn
    return x
    
class LCMReconstruction(Layer):
    '''
    Takes predicted LCM parameters and reconstruct predicted spectra
    '''
    def __init__(self, basis_OFF, basis_DIFF, ppm_array, PPMST, PPMEND, low_vals, high_vals, **kwargs):
        super().__init__(**kwargs)
        self.basis_OFF = tf.constant(basis_OFF, dtype=tf.complex64)
        self.basis_DIFF = tf.constant(basis_DIFF, dtype=tf.complex64)
        self.ppm_array = tf.constant(ppm_array, dtype=tf.complex64)
        self.PPMST = PPMST
        self.PPMEND = PPMEND
        self.low_vals = tf.constant(low_vals, dtype=tf.float32)
        self.high_vals = tf.constant(high_vals, dtype=tf.float32)

    def call(self, pred_params):
        amp = pred_params[:, :27]
        ph = pred_params[:, 27:29]
        lbG = pred_params[:, 29:30]
        lbL = pred_params[:, 30:57]
        fr = pred_params[:, 57:84]
        bl = pred_params[:, 84:96]

        # De-normalize and scale LCM parameters
        amp_OFF = amp[:, :26]
        amp_OFF = amp_OFF * (self.high_vals[:26] - self.low_vals[:26]) + self.low_vals[:26]

        amp_DIFF = amp
        amp_DIFF = amp_DIFF * (self.high_vals[:27] - self.low_vals[:27]) + self.low_vals[:27]

        ph0 = ph[:, 0]
        ph0 = ph0 * (self.high_vals[27] - self.low_vals[27]) + self.low_vals[27]
        ph0 = ph0 * 360.

        ph1 = ph[:, 1]
        ph1 = ph1 * (self.high_vals[28] - self.low_vals[28]) + self.low_vals[28]
        ph1 = ph1 * 10.

        lbG = lbG * (self.high_vals[29] - self.low_vals[29]) + self.low_vals[29]
        lbG = lbG * math.sqrt(5000.)

        lbL_OFF_metab = lbL[:, :18]
        lbL_OFF_metab = lbL_OFF_metab * (self.high_vals[30:48] - self.low_vals[30:48]) + self.low_vals[30:48]
        lbL_OFF_metab = lbL_OFF_metab * 10.

        lbL_DIFF_metab = lbL[:, :18]
        lbL_DIFF_metab = lbL_DIFF_metab * (self.high_vals[30:48] - self.low_vals[30:48]) + self.low_vals[30:48]
        lbL_DIFF_metab = lbL_DIFF_metab * 10.

        lbL_OFF_MM = lbL[:, 18:26]
        lbL_OFF_MM = lbL_OFF_MM * (self.high_vals[48:56] - self.low_vals[48:56]) + self.low_vals[48:56]
        lbL_OFF_MM = lbL_OFF_MM * 50.

        lbL_DIFF_MM = lbL[:, 18:27]
        lbL_DIFF_MM = lbL_DIFF_MM * (self.high_vals[48:57] - self.low_vals[48:57]) + self.low_vals[48:57]
        lbL_DIFF_MM = lbL_DIFF_MM * 50.

        lbL_OFF = tf.concat([lbL_OFF_metab, lbL_OFF_MM], axis=1)
        lbL_DIFF = tf.concat([lbL_DIFF_metab, lbL_DIFF_MM], axis=1)

        fr_OFF_metab = fr[:, :18]
        fr_OFF_metab = fr_OFF_metab * (self.high_vals[57:75] - self.low_vals[57:75]) + self.low_vals[57:75]
        fr_OFF_metab = fr_OFF_metab * (42.577 * 3 * 0.03)

        fr_DIFF_metab = fr[:, :18]
        fr_DIFF_metab = fr_DIFF_metab * (self.high_vals[57:75] - self.low_vals[57:75]) + self.low_vals[57:75]
        fr_DIFF_metab = fr_DIFF_metab * (42.577 * 3 * 0.03)

        fr_OFF_MM = fr[:, 18:26]
        fr_OFF_MM = fr_OFF_MM * (self.high_vals[75:83] - self.low_vals[75:83]) + self.low_vals[75:83]
        fr_OFF_MM = fr_OFF_MM * (42.577 * 3 * 0.05)

        fr_DIFF_MM = fr[:, 18:27]
        fr_DIFF_MM = fr_DIFF_MM * (self.high_vals[75:84] - self.low_vals[75:84]) + self.low_vals[75:84]
        fr_DIFF_MM = fr_DIFF_MM * (42.577 * 3 * 0.05)

        fr_OFF = tf.concat([fr_OFF_metab, fr_OFF_MM], axis=1)
        fr_DIFF = tf.concat([fr_DIFF_metab, fr_DIFF_MM], axis=1)

        bl = bl * (self.high_vals[84:96] - self.low_vals[84:96]) + self.low_vals[84:96]
        bl_OFF = tf.gather(bl, [0, 2, 4, 6, 8, 10], axis=1)
        bl_DIFF = tf.gather(bl, [1, 3, 5, 7, 9, 11], axis=1)

        # Metabolite-specific Lorentzian line-broadening
        time_array = tf.range(0, 2048. * 2) * 0.0005

        win_off = tf.math.exp(tf.expand_dims(-lbL_OFF, axis=-1) * math.pi * time_array)
        win_off = tf.transpose(win_off, perm=[0, 2, 1])
        win_off = tf.cast(win_off, tf.complex64)

        win_diff = tf.math.exp(tf.expand_dims(-lbL_DIFF, axis=-1) * math.pi * time_array)
        win_diff = tf.transpose(win_diff, perm=[0, 2, 1])
        win_diff = tf.cast(win_diff, tf.complex64)

        basis_OFF = tf.multiply(self.basis_OFF, win_off)
        basis_DIFF = tf.multiply(self.basis_DIFF, win_diff)

        # Metabolite-specific frequency shift
        time_array = tf.cast(time_array, tf.complex64)
        win_off = tf.math.exp(2 * math.pi * (tf.complex(0., tf.expand_dims(fr_OFF, axis=-1)) * time_array)) # output_fr in Hz
        win_off = tf.transpose(win_off, perm=[0, 2, 1])

        win_diff = tf.math.exp(2 * math.pi * (tf.complex(0., tf.expand_dims(fr_DIFF, axis=-1)) * time_array)) # in Hz
        win_diff = tf.transpose(win_diff, perm=[0, 2, 1])

        basis_OFF = tf.multiply(basis_OFF, win_off)
        basis_DIFF = tf.multiply(basis_DIFF, win_diff)

        # Linear combination of basis functions
        amp_OFF = tf.cast(amp_OFF, dtype=tf.complex64)
        amp_DIFF = tf.cast(amp_DIFF, dtype=tf.complex64)
        fid_OFF = tf.reduce_sum(tf.multiply(amp_OFF[:, tf.newaxis, :], basis_OFF), axis=-1)
        fid_DIFF = tf.reduce_sum(tf.multiply(amp_DIFF[:, tf.newaxis, :], basis_DIFF), axis=-1)

        # Gaussian line-broadening
        time_array = tf.cast(time_array, tf.float32)
        win = tf.math.exp(-(lbG * math.pi * time_array)**2)
        win = tf.cast(win, tf.complex64)

        fid_OFF = tf.multiply(fid_OFF, win)
        fid_DIFF = tf.multiply(fid_DIFF, win)

        # Phase shift (zero-order)
        ph0 = tf.math.exp(2 * math.pi * (tf.complex(0., ph0/360.))) # output_ph0 in degrees

        fid_OFF = tf.multiply(fid_OFF, ph0)
        fid_DIFF = tf.multiply(fid_DIFF, ph0)

        # FFT
        spec_OFF = tf.signal.fftshift(tf.signal.fft(fid_OFF))
        spec_DIFF = tf.signal.fftshift(tf.signal.fft(fid_DIFF))
        
        # Phase shift (first-order)
        ph1 = tf.math.exp(2 * math.pi * (tf.complex(0., ph1/360.) * self.ppm_array)) # output_ph1 in degrees/PPM

        spec_OFF = tf.multiply(spec_OFF, ph1)
        spec_DIFF = tf.multiply(spec_DIFF, ph1)

        # Crop to fitting range
        start_idx = find_nearest(self.ppm_array, self.PPMST)
        end_idx = find_nearest(self.ppm_array, self.PPMEND)
        nPoints = start_idx - end_idx
        zeros = tf.zeros(nPoints, dtype=tf.int32)

        ppm_range = tf.range(end_idx, start_idx, 1, dtype=tf.int32)
        ppm_range = tf.concat([tf.expand_dims(zeros, -1), tf.expand_dims(ppm_range, -1)], -1)

        spec_OFF = tf.expand_dims (tf.transpose(spec_OFF, perm=[1, 0]), axis=0)
        spec_OFF= tf.transpose ( tf.gather_nd(spec_OFF, ppm_range) )
        spec_DIFF = tf.expand_dims (tf.transpose(spec_DIFF, perm=[1, 0]), axis=0)
        spec_DIFF = tf.transpose ( tf.gather_nd(spec_DIFF, ppm_range) )

        # Add baselines
        x = tf.linspace(-1.0, 1.0, nPoints)
        xx = tf.stack([x**5, x**4, x**3, x**2, x, tf.ones_like(x)], axis=0)

        poly_OFF = tf.linalg.matmul(bl_OFF, xx)
        poly_OFF = tf.cast(poly_OFF, dtype=tf.float32)
        spec_OFF = tf.math.real(spec_OFF) + poly_OFF

        poly_diff = tf.linalg.matmul(bl_DIFF, xx)
        poly_diff = tf.cast(poly_diff, dtype=tf.float32)
        spec_DIFF = tf.math.real(spec_DIFF) + poly_diff

        # Normalize spectra
        scale_OFF = tf.expand_dims(tf.reduce_max(tf.math.abs(tf.math.real(spec_OFF)), axis=1), axis=-1)
        spec_OFF = tf.math.real(spec_OFF) / scale_OFF

        scale_DIFF = tf.expand_dims(tf.reduce_max(tf.math.abs(tf.math.real(spec_DIFF)), axis=1), axis=-1)
        spec_DIFF = tf.math.real(spec_DIFF) / scale_DIFF

        # Concatenating OFF & DIFF
        output_specs = tf.concat([tf.expand_dims(spec_OFF, -1), tf.expand_dims(spec_DIFF, -1)], -1)
        output_specs = tf.cast(output_specs, dtype=tf.float32)

        return output_specs

