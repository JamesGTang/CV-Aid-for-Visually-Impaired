from utils import models, prediction_utils
from utils.load_data import load_data


def main():
    # Testing Params
    batch_size = 64
    model_type = 'V1'  # V1, V2
    weight_path = "saved_ckpt/mobilenet"+model_type+".h5"
    data_dir = "path/to/data/folder"

    # load the right pre-trained model and set the corresponding optimizer
    if model_type == 'V1':
        model, ft_layers = models.mobilenetv1_transfer_learning(num_classes=8)
        model.load_weights(weight_path)
        preprocess = True
    elif model_type == 'V2':
        model, ft_layers = models.mobilenetv2_transfer_learning(num_classes=8)
        model.load_weights(weight_path)
        preprocess = True

    # Get my data loaders
    _, _, test_batches = load_data(batch_size=batch_size, head_dir=data_dir, preprocess=preprocess)
    test_labels = test_batches.classes
    predictions = model.predict_generator(test_batches, steps=29, verbose=2)
    prediction_utils.calculate_test_results_Regression(y_true=test_labels, y_pred=predictions)


if __name__ == '__main__':
    main()
