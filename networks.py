from tensorflow.keras import layers, models, optimizers


def get_expanded_model(input_size, n_actions, lr):
    """Returns short conv model with mask at the end of the network. Network is interpretation of original network
    from papers."""
    screen_input = layers.Input(input_size)
    actions_input = layers.Input(n_actions)

    x = layers.Lambda(lambda x: x / 255.0)(screen_input)

    x = layers.Conv2D(16, (3, 3), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(n_actions)(x)
    x = layers.Multiply()([x, actions_input])

    model = models.Model(inputs=[screen_input, actions_input], outputs=x)
    optimizer = optimizers.RMSprop(lr=lr, rho=0.95, epsilon=0.01, momentum=0.95)
    model.compile(optimizer, loss='mse')
    return model


def get_original_model(input_size, n_actions, lr):
    """Returns short conv model with mask at the end of the network. Copy of network from papers."""
    screen_input = layers.Input(input_size)
    actions_input = layers.Input(n_actions)

    x = layers.Lambda(lambda x: x / 255.0)(screen_input)

    x = layers.Conv2D(16, (8, 8), strides=(4, 4))(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, (4, 4), strides=(2, 2))(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(n_actions)(x)
    x = layers.Multiply()([x, actions_input])

    model = models.Model(inputs=[screen_input, actions_input], outputs=x)
    optimizer = optimizers.RMSprop(lr=lr, rho=0.95, epsilon=0.01, momentum=0.95)
    model.compile(optimizer, loss='mse')
    return model


def get_dense_model(input_size, n_actions, lr):
    state_input = layers.Input(input_size)
    actions_input = layers.Input(n_actions)

    x = layers.Flatten()(state_input)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(n_actions, activation='linear')(x)
    x = layers.Multiply()([x, actions_input])

    model = models.Model(inputs=[state_input, actions_input], outputs=x)
    optimizer = optimizers.RMSprop(lr=lr, momentum=0.95)
    model.compile(optimizer, loss='mse')
    return model
