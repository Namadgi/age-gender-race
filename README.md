# Age Gender Race Estimation
Age Gender Race Estimation is an AI pipeline for detecting the face on the image and estimating the person's age, gender, and race. 
The current branch is an Android mobile application that utilizes the proposed pipeline. 
It uses the [MTCNN](https://github.com/ipazc/mtcnn) (Multi-task Cascaded Convolutional Networks) model for face detection and the [MobileNetV1](https://github.com/jmjeon94/MobileNet-Pytorch) model for estimating the person's age, gender, and race.
Moreover, the current branch provides the PyTorch Mobile version of a model as [SDK](https://github.com/Namadgi/age-gender-race/blob/mobile/model_weights/agre.ptl), which takes an image as input and returns the label or the error code as output.

## App Installation

### Requirements

- Android: Android 7.0 (Nougat) or later
- Minimum 2GB RAM recommended

### Installation Steps

1. Clone this repository to your local machine.
   ```
   git clone -b mobile https://github.com/Namadgi/age-gender-race.git
   ```
2. Open the project in your preferred IDE (Android Studio for Android).
3. Build and run the project on your device or emulator.

![image](https://github.com/Namadgi/face-emotion-recognition/assets/44228198/53013beb-61e6-4372-9c6a-47eaf9d35968)

## SDK Usage

Depending on the OS and platform used, install the framework for PyTorch mobile inference.
To use it on Flutter it is recommended to use the following package: [pytorch_mobile](https://pub.dev/packages/pytorch_mobile).

### Install Package and Create Asset

To use this plugin, add `pytorch_mobile` as a dependency in your `pubspec.yaml` file.
Create a `assets` folder with your pytorch model and labels if needed. Modify `pubspec.yaml` accordingly.

```
assets:
 - assets/models/agre.ptl
```
Run flutter pub get

### Load Model
```
Model customModel = await PyTorchMobile
        .loadModel('assets/models/agre.ptl');
```
### Load Image
Load an image as an RGB BitMap with a (`H, W, C`) shape, where `H` and `W` are height and width of the image, and `C` is color-channel dimension (it always should be 3).
To send the image to the model:

```
_imagePrediction = await _imageModel!.getImagePrediction(
        File(image!.path), 224, 224);
```

The result is the label. The negative values signify one of the errors. Other values stand for emotion class:
```
{
   'code': 0,
   'age': 20,
   'gender': 0,
   'race': 2,
}
```
The value of `code` is one of the following integers `[0, 2, 3, 4]`, where `{0: 'Successful check', 2: 'No face, too small or bad quality', 3: 'Found more than ONE face', 4: 'Face is not centered'}`.

The value of `age` is an integer in the range `[0:100]`. Value of `age` stands for the person's actual age.

The value of `gender` is a binary, where `{0: "Female", 1: "Male"}`.

The value of `race` is an integer, where `{0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Latino / Middle Eastern"}`.

## Credits

This project utilizes the following open-source libraries and pre-trained models:

- [Mobile-App](https://github.com/av-savchenko/face-emotion-recognition): The original repository.
- [MTCNN](https://github.com/ipazc/mtcnn): Multi-task Cascaded Convolutional Networks for face detection.
- [MobileNetV1](https://github.com/jmjeon94/MobileNet-Pytorch): MobileNet neural network architecture.
