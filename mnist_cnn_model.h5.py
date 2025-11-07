# 1. Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
# 2. Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0–255) to (0–1)
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

print("Training samples:", train_images.shape)
print("Testing samples:", test_images.shape)
# 3. Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 digit classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
# 4. Train the model
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)
# 5. Evaluate on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
# 6. Get predictions
predictions = model.predict(test_images)

# Pick five sample images
num_samples = 5
indices = np.random.choice(len(test_images), num_samples, replace=False)

plt.figure(figsize=(10, 3))
for i, idx in enumerate(indices):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(test_images[idx].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[idx])}")
    plt.axis('off')
plt.show()
model.save("mnist_cnn_model.h5")
