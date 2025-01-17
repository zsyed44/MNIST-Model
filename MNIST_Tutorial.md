# **MNIST Handwritten Digit Classifier - Workshop Guide**

## **Abstract**
This document, written by **Zain Syed**, the VP Tech at WE AutoPilot, will guide you through building (potentially) your first **Machine Learning model!** We will be using the MNIST dataset to make an AI tool to recognize handwritten numbers. The principle is, by submitting thousands of handwritten images, we can teach our computer to recognize patterns, making it capable of recognizing numbers

### **Why is this important?**
Well, although it may seem like a trivial task to read handwritten numbers (unless it’s my handwriting), for a computer it is really huge. If I get a calculator, or a computer to type out the number “8”, they do not actually understand what 8 is, or what it represents. It is merely a symbol among others to a computer. If we can train a computer to **visually interpret and understand** a number, that is incredible news! It is like the natural next step when it comes to computer programming. Teaching a computer to visually interpret numbers is a major advancement and is foundational for applications such as **autonomous driving, AI handwriting recognition, and more**.

---

## **Stage 1: Getting Started - Installing Dependencies**

### **Prerequisites:**
Ensure you have the following installed:
- **Python** (with Pip)
- **IDE** (Recommended: VSCode)

### **Installing Required Libraries**
Run the following command in your terminal:
```sh
pip install tensorflow numpy matplotlib seaborn
```
If you encounter issues, refer to the [FAQ](#faqs-and-common-problems) or ask a workshop mentor.

### **Creating a Python File**
Create a new Python file (e.g., `main.py`) and import the required libraries:
```python
import tensorflow
import numpy
import matplotlib.pyplot
```

**OR**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```
It all depends on how you wish to define your namespaces!

---

## **Stage 2: Data Preprocessing**
Preprocessing ensures our dataset is **consistent** and **optimized** for training.

### **Loading the MNIST Dataset**
```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
**Dataset Breakdown:**
| Variable | Description |
|----------|-------------|
| `x_train` | 60,000 28×28 pixel images for training |
| `y_train` | Labels (digits 0-9) for `x_train` |
| `x_test` | 10,000 images for testing |
| `y_test` | Labels for `x_test` |

### **Optional: Visualizing the Data**
```python
import seaborn as sns
sns.countplot(x=y_train)
```

### **Optional: Checking for Missing Values**
```python
print(np.isnan(x_train).any())
print(np.isnan(x_test).any())
```
Ideally here we are looking for both to print out the word "False"

---

## **Stage 3: Data Normalization & Reshaping**
TensorFlow expects **4D input** for CNNs (`(batch_size, height, width, channels)`). We reshape and normalize our images:

```python
input_shape = (28, 28, 1)  # Define input shape (1 channel for grayscale)

# Reshape images and normalize pixel values
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test.astype('float32') / 255.0

```

### **Encoding Labels (One-Hot Encoding)**
```python
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
```

---

## **Stage 4: Building the Convolutional Neural Network (CNN)**

### **Defining Model Parameters**
```python
batch_size = 64  # Number of images processed at once
epochs = 5       # Number of training cycles
num_classes = 10 # Digits (0-9)
```

### **Building the CNN Model**
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

### **Compiling the Model**
```python
model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### **Callback for Early Stopping**
```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.995:
            print("\nReached 99.5% accuracy, stopping training!")
            self.model.stop_training = True

callbacks = myCallback()
```

### **Training the Model**
```python
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[callbacks])
```

---

## **Stage 5: Saving and Running the Model**
### **Save the Model to an `.h5` File**
```python
model.save("mnist_model.h5")
model.summary()
```

### **Run the Training Script**
```sh
python main.py
```
Replace `main.py` with your script name.

---

## **FAQs and Common Problems**
### **Error: Could not install packages due to OSError: [WinError 2]**
**Solution:**
1. Ensure Python is installed.
2. Add Python to your PATH:
   - Press `Win + R`, type `sysdm.cpl`, and press Enter.
   - Navigate to: **Advanced > Environment Variables > Path**.
   - Click **New** and add:
     ```
     C:\Python310\
     C:\Python310\Scripts
     ```
   - Restart your computer and run:
     ```sh
     where python
     ```
   - If Python is detected, the issue should be resolved.

---

## **What's Next?**
🎉 **Congratulations!** You've successfully trained an MNIST digit recognition model! Explore further by:
- Testing different **hyperparameters**.
- Improving the **model architecture**.
- Using **data augmentation**.
- Running the **GUI tool** to interact with your model.

📌 **For additional help, visit the workshop mentors or the GitHub repository.** 🚀

---

## **Images of potential GUIs in the program**

![image](https://github.com/user-attachments/assets/271a8df1-2525-4c0f-a167-fa4e0b9611cd)
![image](https://github.com/user-attachments/assets/736c3583-9b79-42e4-a70e-c75b99971061)
![image](https://github.com/user-attachments/assets/8402f4e0-4394-4c63-bdd0-e9345ada95c8)
![image](https://github.com/user-attachments/assets/1eaddff2-ba33-4392-a804-fe67bbd51900)