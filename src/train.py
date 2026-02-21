from tensorflow import keras

def build_callbacks(patience=10):
    early_stop = keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=5e-7,
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )
    return [early_stop]


def train_model(model, X_input, X_filled, batch_size=1, epochs=2500, callbacks=None):
    '''
    Per-instance test-time training/adaptation function
    '''
    history = model.fit(
        x=[X_input, X_filled],
        y=None,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        shuffle=False,
        callbacks=callbacks
    )
    return history