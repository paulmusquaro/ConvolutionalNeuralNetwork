# ConvolutionalNeuralNetwork

###### Convolutional Neural Networks (CNN) & Residual Neural Network (ResNet)

This project involves building and comparing neural network architectures for classifying products from the Fashion MNIST dataset. Specifically, the focus is on the development of Convolutional Neural Networks (CNNs) and Residual Neural Networks (ResNets). The goal is to explore different architectures, optimize their performance, and identify the best-performing model based on classification accuracy.

## Objectives

1. **Convolutional Neural Network (CNN):**
   - Develop a CNN using Keras to classify images from the Fashion MNIST dataset.
   - Optimize the architecture and parameters to achieve high classification accuracy.
   - Compare the performance of the CNN with a baseline Multilayer Perceptron (MLP).

2. **Residual Neural Network (ResNet):**
   - Implement a CNN architecture based on VGG16 as a convolutional base.
   - Utilize feature extraction and fine-tuning to enhance performance.
   - Compare its performance to an MLP and draw conclusions.

## Dataset

The project uses the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, which contains 70,000 grayscale images of 10 clothing categories, each represented as a 28x28 pixel image.

- **Classes:**
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

---

## Conda (Setup and Environment)

To make the project reproducible and ensure smooth package management, this project uses Conda as a package and environment manager. Below are the steps to set up the environment:


1. **Install Conda**:
If you haven't installed Conda yet, you can download it from the official [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) websites. Anaconda is a larger distribution with more pre-installed packages, while Miniconda is a smaller, minimal version. Choose whichever suits your needs.

2. **Create a new environment:** Open your terminal and run the following command to create a new Conda environment with Python 3.7:

    ```bash
    conda create --name new_conda_env python=3.7
    ```

3. **Activate the environment:** Once the environment is created, activate it by running:

    ```bash
    conda activate new_conda_env
    ```

4. **Install required packages (Jupyter, NumPy, MatPlotLib, Pandas, Scikit-Learn, Tensorflow and Keras)**

    ```bash
    conda install jupyter numpy matplotlib pandas scikit-learn tensorflow keras
    ```

5. **Run Jupyter Notebook**

    ```bash
    jupyter notebook
    ```

---

## Methodology

### 1. Convolutional Neural Network (CNN)
- **Architecture:**
  - Multiple Conv2D layers with ReLU activation and MaxPooling layers.
  - Dropout layers to prevent overfitting.
  - Dense layers for classification with a softmax activation function.
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Key Results:**
  - Achieved **91% validation accuracy** after 10 epochs.
  - The CNN outperformed the MLP in validation accuracy.

### 2. Residual Neural Network (ResNet) with VGG16
- **Architecture:**
  - VGG16 as a non-trainable convolutional base.
  - Fully connected Dense layers added on top.
  - Data augmentation applied for better generalization.
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Key Results:**
  - Achieved **81% validation accuracy** after 40 epochs without fine-tuning.
  - Performance improved with data augmentation but remained lower than the CNN.

## Insights and Conclusions

- The CNN demonstrated better classification performance compared to both the MLP and the ResNet (based on VGG16).
- **Best Performance:**
  - CNN: **91% validation accuracy**
  - ResNet (VGG16): Performed well but fell short compared to the simpler CNN model.
- Fine-tuning and additional data augmentation could potentially improve ResNet performance further.

## Future Work

- Experiment with additional architectures, such as deeper CNNs or different pre-trained models (e.g., ResNet50 or Inception).
- Explore techniques like transfer learning, hyperparameter tuning, and increased training epochs.
- Analyze model performance on edge cases and under different data augmentation strategies.