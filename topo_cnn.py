import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, ReLU, GlobalAveragePooling2D, Concatenate

from tensorflow.keras.applications import DenseNet121, MobileNetV2, InceptionV3


def build_cnn_tda_model(input_shape_img, tda_dim, num_classes, base_model_name="DenseNet121"):
    base_models = {
        "DenseNet121": DenseNet121,
        "MobileNetV2": MobileNetV2,
        "InceptionV3": InceptionV3
    }

    if base_model_name not in base_models:
        raise ValueError(f"Unsupported base model: {base_model_name}. Choose from {list(base_models.keys())}")

    base_model_fn = base_models[base_model_name]
    base_model = base_model_fn(include_top=False, weights="imagenet", input_shape=input_shape_img)
    base_model.trainable = False

    # Image input
    img_input = Input(shape=input_shape_img, name="img_input")
    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    cnn_feat = Dense(128, activation='relu')(x)

    # TDA input
    tda_input = Input(shape=(tda_dim,), name="tda_input")
    y = Dense(512, activation='relu')(tda_input)
    y = Dense(256, activation='relu')(y)
    y = Dense(128, activation='relu')(y)
    tda_feat = y

    # Fusion and final classifier
    combined = Concatenate()([cnn_feat, tda_feat])
    z = Dense(64)(combined)
    z = ReLU()(z)
    output = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[img_input, tda_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model