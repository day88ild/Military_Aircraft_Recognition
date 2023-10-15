# Military Aircraft Recognition Project

This repository contains the code and resources for a machine learning project aimed at recognizing and classifying military aircraft in images from drones. Below is an overview of the project structure and its various components.

## Project Description

- **Task**: Object Recognition and Classification Task
- **Goal**: Develop a machine learning model for recognizing and classifying military aircraft in images.
- **Task Duration**: 1 week

## Project Structure

- **data**: This directory contains the dataset used for the project.
- **empty_images**: Images without any aircraft, used for model training and testing.
- **runs**: Contains model files, results, and logs.

### Data Analysis

- `data_analysis.ipynb`: Jupyter Notebook for dataset analysis, including dataset structure, distribution, and visualization.

### Data Preparation

- `data_preparation.ipynb`: Jupyter Notebook for dataset preparation, including data augmentation and splitting into training and testing sets.

### Model Training

- `first_model_training.ipynb`: Jupyter Notebook for training the initial classification model.
- `Yolo_model_nano_training.ipynb`: Jupyter Notebook for training a YOLOv8-based model (nano version).
- `Yolo_model_small_training.ipynb`: Jupyter Notebook for training a YOLOv8-based model (small version).
- `RCNN_model_training.ipynb`: Jupyter Notebook for training a Faster R-CNN model.

### Empty Images

- `empty_images`: A directory containing images without any aircraft. Used for training and testing the models.

### Model Results

- `runs`: Contains model weights, results, and logs.
  - **clf**: Classification model results.
  - **detect**: Object detection model results.
    - **train**: Training results.
    - **train2**: Additional training results.

### Model Weights and Logs

- `weights`: Contains model weights.
- `results.csv`: A CSV file containing results of model evaluations.

### Other Resources

- `Military_Aircraft_Recognition_task_description.md`: The task description document.
- `Trainee ML Engineer Vacancy - Object Recognition and Classification Task.pdf`: The initial task description document.

## Model Performance

### Classification Model (First Model)

- Accuracy: 0.918
- Precision: 0.9262
- Recall: 0.9147
- F1 Score: 0.9204

### Object Detection Models (Yolov8nano)

- Precision: 0.87081
- Recall: 0.85034
- F1 Score: 0.8605

- More model results can be found in the `runs` directory and also they are represented in Yolo_model_nano_training.ipynb.

### Object Detection Models (Yolov8small)

- There is no trained model but if you have enough cpu you can train your own with this script Yolo_model_small_training.ipynb

### Object Detection Models (Faster RCNN)

- There is no trained model but if you have enough cpu you can train your own with this script RCNN_model_training.ipynb



## Future Integration

In the future, these models could be integrated into an automatic object recognition system for military applications, as described in the [project documentation](Military_Aircraft_Recognition_task_description.md).

## How to Use

You can refer to the individual Jupyter Notebooks for details on data analysis, data preparation, and model training. You can download data in the script in data_analysis.ipynb and prepare data for Yolo and first clf model in data_preporation.ipynb. The `runs` directory contains model weights and logs.

Feel free to explore and use the provided resources for your object recognition and classification tasks.

## License

This project is open-source and is provided under the [license](LICENSE) file.

