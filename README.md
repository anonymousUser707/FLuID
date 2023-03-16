# Invariant Dropout

This project contains the Invariant Dropout implementation for CIFAR10, FEMNIST, and Shakespeare
Also implements baseline techniques Random and Ordered Dropout

This project is implemented using the [Flower framework v0.18.0](https://github.com/adap/flower). Also uses Flower's [Android example](https://flower.dev/blog/2021-12-15-federated-learning-on-android-devices-with-flower/) as a implementation basis.

NOTE: Technique requires running a minimum of 2 mobile/handheld clients for training

## Requirements and Installation

The project requires **Python >=3.7**, and uses **Gradle** to build the client mobile app. 
The mobile application has been verified to be working for devices with at least Android 9 (sdk 28)

The primary Project dependencies are project dependencies `tensorflow` and `flwr` (v0.18.0)

```shell
pip install flwr==0.18.0
pip install tensorflow
```

## Datasets
In this and following sections replace instances of`<Dataset>` with the dataset you are running the experiment with (CIFAR10, FEMNIST, or Shakespeare). These datasets are all publicly availble for download.
 
The Shakespeare and FEMNIST datasets can be obtained from the [official LEAF Repo] (https://github.com/TalwalkarLab/leaf). 
Please follow the instructions there to download the LEAF datasets there
   - 5 example user datasets are included for each dataset in `<Dataset>/client/app/src/main/assets/data`

The [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) Implementation uses the dataset partitioned by and provided Flower and can be downloaded from this [link](https://www.dropbox.com/s/coeixr4kh8ljw6o/cifar10.zip?dl=1)

**Setup instructions below outline the general steps to run Invariant Dropout, for dataset specific processing instructions please refer to the README in each dataset subforlder.**

## Client Application Setup

There are two steps to setting up the Andoird App for running the project
### 1. Generating `.tflite` models
**NOTE**:  **No additional** model definition needed to be generated to run the current code.
  - All 6 required model definitions to run the current code `p=[1.0, 0.95, 0.85, 0.75, 0.65, 0.5]` are included in `<Dataset>/client/app/src/main/assets/model`
  - To modify the model, or add new sub-model sizes, you would need to generate `.tflite` files for all sub-model sizes that you wish to run, plus the full model size `p=1.0`
     
To generate more model definitions for sub-model sizes: 
1. Define the modle in `<Dataset>/tflite_convertor/convert_to_tflite.py`.  
2. Vary the `p` variable to create models of different sizes (1.0 => full model, 0.5 = model with half the size)
3. Execute the script, and models will be created un the `tflite_convertor/tflite_model` folder
  ```shell
  python convert_to_tflite.py
  ```
4. Rename each file as `<p>_<original file name>.tflite`. 
   - For example, a the `train_head.tflite` file create with `p=0.75` would be renamed as `0.75_train_head.tflite`
6. Add files to `<Dataset>/client/app/src/main/assets/model` directory

### 2. Add data to the assets folder
   - Follow the `Processing the Dataset` instructions in the README files for each dataset subfolder
   - Add the downloaded datasets to `<Dataset>/client/app/src/main/assets/data` directory


##  Building Client Application

To build and install the application on an Android Deivce, first do the following:
- Enable `Developer Mode` and `USB debugging` on the Android Device
- Connect a mobile device wirelessly or using an USB cord

Then there are a few options to build the application:

1. Using Android Studio
    - In Android Studio open the project at `<Dataset>/client/`
    - Use the `Run App` function on Android Studio to build and install the application
2. Using Gradlew
    - In terminal go to `<Dataset>/client/`
    - Run `gradlew installDebug` (if on Windows)
        - If on Max or Linux run `./gradlew installDebug`
        - This will build and install the app on your device
        
Note: in case of an `SDK location not found` Error, create a file `local.properties` file in`<Dataset>/client/` with the following line:
```shell
sdk.dir=<sdk dir path>
```
where <sdk.dir path> is the path of where your Android SDK is installed.


## Server setup 

1. Each dataset subfolder contains 2 main files
   1. `fedDrop<dataset>_android.py`
      - This is the actual implementation of he dropout methods
      - You can select which Dropout technique to run in the function `configure_fit` simply change line with the desired dropout method
      ```shell 
      fit_ins_drop = FitIns(self.drop_rand(parameters, self.p_val, [0,3], 10, client.cid), config_drop)
      ```
      - (`drop_rand` => random dropout, `drop_order` => ordered dropout, `drop_dynamic` => Invariant dropout)
   2. `server.py`
      - This is a typical Flower framework server script. 
      - Youc an specify the number of clients to run, the server's ip address, and number of rounds to run
         - A reminder that the technique requires a minimum of 2 clients for training
         

## Run Federated Dropout on Android Clients

1. To start the server, simply open a terminal at the subfolder of the dataset, and run 
```shell
python server.py
```
2. Open the app corresponding to the dataset on you phone
3. When the app runs add the client ID (between 1-10), the IP and port of your server, and press `Load Dataset`. This will load the local dataset in memory. Then press `Setup Connection Channel` which will establish connection with the server. Finally, press `Train Federated!` which will start the federated training. 


  




