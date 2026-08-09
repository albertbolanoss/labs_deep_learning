# Deep Learning Samples

## Prerequisites

### Run from Google Colab.

- Google Colab runtime.
- Google Drive Folder (get the GOOGLE_DRIVE_FOLDER_ID and replace in 01_training_model.ipynb).
- Kaggle API Creadentials (add KAGGLE_USERNAME and KAGGLE_KEY as Colab Secrets).
- Download the pre-trained models form [Here](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API).
- Download Yolo weigth model from [Here](https://pjreddie.com/darknet/yolo/).

### Run from Visual Studio Code.

- Install [UV](https://docs.astral.sh/uv/getting-started/installation/)
- Install Visual studio Python Extension and Jupyter Extension by Microsoft provider.


## Virtual environment

```sh
uv venv --python 3.14
```

##  Install dependencies in local

```sh
uv pip install tensorflow matplotlib google-api-python-client google-auth-oauthlib ipykernel
```