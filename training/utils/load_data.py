from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras
import tensorflow.keras.applications.mobilenet


def load_data(batch_size, head_dir, preprocess=True, class_mode="categorical"):

    if preprocess:
        preprocess_fn = tensorflow.keras.applications.mobilenet_v2.preprocess_input
    else:
        preprocess_fn=None

    train_path = head_dir+"train"
    train_batches = ImageDataGenerator(preprocessing_function=preprocess_fn).\
        flow_from_directory(train_path, target_size=(224, 224), batch_size=batch_size, shuffle=True, class_mode=class_mode)

    val_path = head_dir+"val"
    val_batches = ImageDataGenerator(preprocessing_function=preprocess_fn).\
        flow_from_directory(val_path, target_size=(224, 224), batch_size=batch_size, shuffle=True, class_mode=class_mode)

    # new_test - steps=29 and batch_size=32, test - steps=20 and batch_size=57, unknown_test - steps=7 batch_size=28
    test_path = head_dir+"test"
    test_batches = ImageDataGenerator(preprocessing_function=preprocess_fn).\
        flow_from_directory(test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False, class_mode=class_mode)

    return train_batches, val_batches, test_batches
