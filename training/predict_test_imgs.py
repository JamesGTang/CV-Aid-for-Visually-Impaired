from utils.prediction_utils import predict_single_image
from utils import models
import glob


def main():
    # Testing Params
    all_fnames = glob.glob("test_imgs/*.png") + glob.glob("test_imgs/*.jpg")
    model_type = 'V1'
    weight_path = "saved_ckpt/mobilenet"+model_type+".h5"

    # load the right pre-trained model and set the corresponding optimizer
    if model_type == 'V1':
        model, ft_layers = models.mobilenetv1_transfer_learning(num_classes=8)
        model.load_weights(weight_path)
    elif model_type == 'V2':
        model, ft_layers = models.mobilenetv2_transfer_learning(num_classes=8)
        model.load_weights(weight_path)

    for fname in all_fnames:
        print(fname, predict_single_image(model, file=fname, model_type=model_type)[0][0]+1)


if __name__ == '__main__':
    main()
