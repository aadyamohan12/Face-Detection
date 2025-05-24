import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt  # Import Matplotlib for graph plotting

# Step 1: Load and Preprocess Data
train_dir = r"C:\Users\Aadya\OneDrive\Desktop\final\final\train" # Replace with path to training dataset
val_dir = r"C:\Users\Aadya\OneDrive\Desktop\final\final\val" # Replace with path to validation dataset

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(600, 600),
                                                    batch_size=32, class_mode='binary')

val_generator = val_datagen.flow_from_directory(val_dir, target_size=(600, 600),
                                                batch_size=32, class_mode='binary')

# Step 2: Build the Model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(600, 600, 3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

spoof_model = models.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

spoof_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model and Save History
history = spoof_model.fit(train_generator, steps_per_epoch=train_generator.samples//train_generator.batch_size,
                          epochs=10, validation_data=val_generator,
                          validation_steps=val_generator.samples//val_generator.batch_size)

# Save the fine-tuned model for future use
spoof_model.save('fine_tuned_spoof_model_600x600.h5')

# Plot Training and Validation Accuracy and Loss
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_training_history(history)

# Step 4: Predict Using the Fine-tuned Model
spoof_model = models.load_model('fine_tuned_spoof_model_600x600.h5')

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (600, 600))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = spoof_model.predict(img_array)
    real_prob = prediction[0][0]
    fake_prob = 1 - real_prob
    return real_prob, fake_prob

def display_result(img_path):
    real_prob, fake_prob = predict_image(img_path)
    
    print(f"Real Probability: {real_prob * 100:.2f}%")
    print(f"Fake Probability: {fake_prob * 100:.2f}%")
    
    img = cv2.imread(img_path)
    cv2.imshow('Input Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = r"C:\Users\Aadya\OneDrive\Desktop\final\final\val\fake\easy_1_1110.jpg" # Replace with the path to the image you want to test
    display_result(img_path)
