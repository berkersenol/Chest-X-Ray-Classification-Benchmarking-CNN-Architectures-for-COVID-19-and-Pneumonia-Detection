# Dataset

This project uses the **COVID-19, Pneumonia, and Normal Chest X-Ray PA Dataset** from Kaggle.

## Download Instructions

1. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json`.

3. Download the dataset:
   ```bash
   kaggle datasets download -d amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset
   unzip covid19-pneumonia-normal-chest-xray-pa-dataset.zip -d COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset
   ```

4. Ensure the extracted folder is placed in the project root:
   ```
   covid-xray-cnn-comparison/
   └── COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset/
       ├── covid/
       ├── normal/
       └── pneumonia/
   ```

The `metadata.csv` file in this directory contains the image paths and class labels used for training.
