# Face-Recognition-with-CNN-and-Transfer-Learning
Face recognition has become one of the most crucial technologies in the world, finding applications in security surveillance, identity verification, and beyond. Traditional face recognition systems relied on handcrafted features and statistical methods, but they often struggled with challenges like variations in pose, lighting, expressions, and occlusions. However, deep learning—especially Convolutional Neural Networks (CNNs)—has completely transformed the field. By enabling automated feature extraction, CNNs have significantly improved recognition accuracy.
That said, training a CNN from scratch for face recognition comes with major drawbacks. It requires an enormous dataset and high computational power, both of which are often inaccessible to many projects due to resource and time constraints. This is where transfer learning comes in. Instead of building a model from the ground up, we can use pre-trained models like VGG and ResNet. These models have already been trained on ImageNet, a massive dataset with over 1.2 million labeled images across 1,000 categories.


DEEP LEARNING MODEL


Deep learning is a neural network which can perform complex computations when the data is especially large. Neural network is made of nodes which are connected to each other in the layer. It is generally classified into three layers.
The key parts are:
1/ The input layer


It receives the raw data or features to be processed.
The number of neurons is equal to the number of the input features.
2/ The hidden layer

Intermediate layers that are used to perform more complex tasks where the model has to learn levels of representation of the data.
Consists of neurons that contain activation functions such as ReLU, Sigmod, and Tanh.
3/ The output layer
It is the layer which is used for the classification purpose.
Final output is created, which can be an example of class probabilities or number prediction.
Types of Deep Learning Models
Convolutional Neural Networks (CNNs)
CNNs are specialized for processing grid-like data, such as images, and have shown outstanding performance in image classification, object detection, and face recognition.
How CNNs Work


Convolutional Layer – Applies filters to the input image to detect edges, textures, and patterns.
Pooling Layer – Reduces the size of the feature maps while preserving important information (e.g., max pooling or average pooling).
Fully Connected Layer – Combines all extracted features and connects to a dense layer for the final classification or prediction step.

VGG (Visual Geometry Group) Network


VGG is a deep CNN architecture with 16 or 19 layers (VGG-16, VGG-19). It is widely used for object detection and image recognition, consistently outperforming benchmarks beyond ImageNet. Despite newer architectures, VGG remains relevant due to its strong performance.
Input Format: Accepts 224×224 RGB images, normalizing pixel values based on the mean RGB values of the training dataset.
Filters: Uses small (3×3) and (1×1) filters with fixed parameters.
Structure: Contains fully connected layers and different variations like VGG-11 (8 convolutional + 3 fully connected layers), VGG-16 (13 convolutional + 3 fully connected layers), and VGG-19 (16 convolutional + 3 fully connected layers). Additionally, it includes five pooling layers that are proportionally distributed with convolutional layers.

VGG-16


VGG-16, developed by the Visual Geometry Group at the University of Oxford, is a CNN with 13 convolutional and 3 fully connected layers. Its relatively simple yet powerful architecture makes it highly effective for tasks like image classification and object recognition, as it learns hierarchical visual features efficiently.
In the 2024 ImageNet competition, where models were tasked with classifying images into 1,000 categories and detecting objects in 200 classes, VGG-16 demonstrated strong performance. Despite newer architectures, it remains widely used due to its versatility and reliability.
  
