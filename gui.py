import tkinter
from tkinter import Canvas
from PIL import Image, ImageGrab
import numpy
import tensorflow
import matplotlib.pyplot as plt

def run_gui(model):
    # GUI setup
    root = tkinter.Tk()
    root.title("MNIST DIGIT CLASSIFIER")

    canvas = Canvas(root, width=300, height=300, bg='white')
    canvas.pack()

    def preprocess(img):
        # resize + scale
        img = img.resize((28, 28)).convert('L')
        img_array = numpy.array(img)
        img_array = 255 - img_array # invert black and white
        img_array = img_array / 255.0 # normalize
        img_array = img_array.reshape(1, 28, 28, 1) # reshape to match model input
        return img_array

    def predict_digit():
        # capture canvas content
        x0 = root.winfo_rootx() + canvas.winfo_x()
        y0 = root.winfo_rooty() + canvas.winfo_y()
        x1 = x0 + canvas.winfo_width()
        y1 = y0 + canvas.winfo_height()
        img = ImageGrab.grab().crop((x0, y0, x1, y1))

        # preprocess and predict
        img_array = preprocess(img)

        plt.imshow(img_array[0, :, :, 0], cmap='gray')
        plt.show()

        # plt.imshow(img_array)
        prediction = model.predict(img_array)
        digit = numpy.argmax(prediction)
        confidence = numpy.max(prediction)

        # display result
        result_label.config(text=f'Prediction: {digit}\nConfidence: {confidence:.2f}')

    def clear_canvas():
        canvas.delete('all')

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill='black')


    canvas.bind("<B1-Motion>", paint)

    btn_predict = tkinter.Button(root, text="Predict", command=predict_digit)
    btn_predict.pack()

    btn_clear = tkinter.Button(root, text="Clear", command=clear_canvas)
    btn_clear.pack()

    result_label = tkinter.Label(root, text="", font=("Helvetica", 16))
    result_label.pack()

    root.mainloop()

def main():
    mnist_dataset = tensorflow.keras.datasets.mnist
    _, (test_images, test_labels) = mnist_dataset.load_data()

    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
    test_images = test_images / 255.0
    test_labels = tensorflow.one_hot(test_labels.astype(numpy.int32), depth=10)

    model = tensorflow.keras.models.load_model('mnist_model.keras')

    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f'Test loss: {test_loss:.3f}, Test accuracy: {test_accuracy:.3f}')
    run_gui(model)

if __name__ == '__main__':
    main()

