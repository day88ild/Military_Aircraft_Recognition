# Military Aircraft Recognition Project

## Project Goal
The primary objective of this project is to develop a machine learning model for detecting and classifying military aircraft in images captured by drones. This project aims to contribute to the development of an automatic object recognition system that can be utilized in military applications.

## Model Building Roadmap

### First Model
1. **Objective**: The initial model aims to classify military aircraft in images. This model will serve as the baseline to establish the minimum Bayes error, which will be used as a performance benchmark for future models.
   
2. **Data Preparation**: Data preparation will involve image cropping to save images with only Aircrafts on them.
   
3. **Model Architecture**:
   - Transfer Learning: A pre-trained deep learning model will be employed to speed up model training.
   
4. **Results**:
   - Accuracy: 0.918
   - Precision: 0.9262
   - Recall: 0.9147
   - F1 Score: 0.9204

   *Note*: Further improvements can be made by using data augmentation and allowing for longer training times.

### Second Model
1. **Objective**: The second model will be based on YOLO (You Only Look Once) architecture, specifically YOLOv8. This model will be designed for object detection in images.

2. **Data Preparation**:
   - Train-Test Split: The dataset will be divided into my training and testing sets. (As data analysis showed us test and train distributions are similar)
   - Augmentation: Data augmentation will include horizontal and vertical flips.
   - Including Images without Aircraft: Images without military aircraft will be added to help the model differentiate between empty images and images containing aircraft. (Data set does not contain any images without aircrafts)

3. **Model Architecture**:
   - YOLOv8: The YOLOv8 architecture, available in the open-source tool from [Ultralytics](https://github.com/ultralytics). We will use two types of architectures: nano and small.
   
4. **Results Yolov8n (nano)**: You can find more results in Yolo_model_nano_training.ipynb.
   - Precision: 0.87081
   - Recall: 0.85034
   - F1 Score: 0.8605

   *Note*: Larger YOLO models may improve results, but resource limitations were encountered during training. I tried to train Yolov8s (small) on Colab but due to limitations of time I did not manage to get any results. Train Yolov8s for one epoch takes up to 5.5 hours while for Yolov8n it is 0.5 hour. You can find notebook for training on Colab in the repository Yolo_modle_small_training.ipynb.

### Third Model
1. **Objective**: The third model will be based on Faster R-CNN (Region-based Convolutional Neural Network) for object detection.

2. **Data Preparation**:
   - Data loader: I created custom data loader to load data from data_yolo.

3. **Model Architecture**:
   - Faster R-CNN: The model will be implemented using PyTorch and torchvision.

4. **Results**:
   - Resource Limitations: Due to hardware limitations, full training could not be performed. I also tried to train the model on Colab but even cpu of free version of Colab wos not enough. However, it is anticipated that with sufficient resources, this model has potential for better performance. You can find notebook for training in the repository RCNN_model_training.ipynb.

## Integration into Automatic Object Recognition System
In the long-term vision with more time and resourses, these models could be trained more and integrated into an automatic object recognition system used in military applications. The integration would involve:

1. Deployment: Deploying the trained models on powerful hardware for real-time object recognition.
   
2. Data Sources: Integrating with remote sensing, satellites, or drone data sources to continuously feed the system with new imagery.

3. Real-time Alerts: Implementing an alerting mechanism that can quickly notify relevant personnel when potential military aircraft are detected.

4. Continuous Improvement: Regularly updating the models and algorithms to adapt to changing conditions and improve accuracy. Getting valuable feedback from millitary experts.

