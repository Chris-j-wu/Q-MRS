import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from CvT import CvTBlock, LCMReconstruction


def build_stage(x, embed_dim, num_heads, ff_dim, num_layers,
              conv_kernel, conv_stride):
    '''
    CvT uses a multi-stage structure. A stage comprises convoltuional token embedding followed by a stack of CvT blocks
    '''
    # Convoltuional token embedding
    x = Conv1D(embed_dim,
                      kernel_size=conv_kernel,
                      strides=conv_stride,
                      padding="valid")(x)
    x = LayerNormalization()(x)

    for _ in range(num_layers):
        x = CvTBlock(x, embed_dim, num_heads, ff_dim)

    return x

def MultiHeadMLP(x, hu=128):
    shared_fc = Dense(512, activation='relu')(x)

    amp_fc = Dense(hu, activation='relu')(shared_fc)
    amp_out = Dense(27, activation='softplus', name='output_amp')(amp_fc)

    global_param_fc= Dense(hu, activation='relu')(shared_fc)
    global_param = Dense(3, name='output_global_param')(global_param_fc)
    ph = Lambda(lambda x: x[:, 0:2])(global_param)
    lbG = Lambda(lambda x: x[:, 2:])(global_param)
    ph_out = Activation('tanh', name='output_ph')(ph)
    lbG_out = Activation('sigmoid', name='output_lbG')(lbG)

    lbL_fc = layers.Dense(hu, activation='relu')(shared_fc)
    lbL_out = Dense(27, activation='sigmoid', name='output_lbL')(lbL_fc)

    fr_fc = Dense(hu, activation='relu')(shared_fc)
    fr_out = Dense(27, activation='tanh', name='output_fr')(fr_fc)

    bl_fc = Dense(hu, activation='relu')(shared_fc)
    bl_out = Dense(12, activation='linear', name='output_bl')(bl_fc)

    output_param = tf.concat([amp_out, ph_out, lbG_out, lbL_out, fr_out, bl_out], 1)

    return output_param

def build_model(input_shape, input_zf_shape, basisSet, PPMST, PPMEND, low_vals, high_vals, model_weights_path=None):

    inp = Input(shape=input_shape)
    inp_zf = Input(shape=input_zf_shape)

    x = build_stage(
        x=inp,
        embed_dim=32,
        num_heads=2,
        ff_dim=32 * 4,
        num_layers=2,
        conv_kernel=7,
        conv_stride=4,
    )

    x = build_stage(
        x=x,
        embed_dim=32,
        num_heads=4,
        ff_dim=32 * 4,
        num_layers=2,
        conv_kernel=3,
        conv_stride=2,
    )

    x = build_stage(
        x=x,
        embed_dim=32,
        num_heads=6,
        ff_dim=32 * 4,
        num_layers=4,
        conv_kernel=3,
        conv_stride=2,
    )

    x = Flatten()(x)

    output_params = MultiHeadMLP(x)
    output_specs = LCMReconstruction(basisSet.fids_zf_OFF, basisSet.fids_zf_DIFF, basisSet.ppm_array_zf, PPMST, PPMEND, low_vals, high_vals)(output_params)

    model = Model(inputs=[inp, inp_zf],
                              outputs=[output_params])
    
    # Load pre-trained weights if provided
    if model_weights_path is not None:
        model.load_weights(model_weights_path)

    model = Model(inputs = model.inputs, outputs=[model.output, output_specs])

    return model