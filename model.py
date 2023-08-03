import tensorflow as tf
from sklearn.model_selection import train_test_split
import load_dataset
images, labels = load_dataset.load_data(load_dataset.dataset1)
x_test = load_dataset.load_test_data(load_dataset.test_data)
# Split the data into train, validation, and test sets
x_train, x_1, y_train, y_1 = train_test_split(images, labels, test_size=0.3, random_state=42)
x_test1, x_val, y_test1, y_val = train_test_split(x_1, y_1, test_size=0.33, random_state=42)

# Build your model (replace this with your actual model architecture)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='linear')  # Linear activation for regression task
])

# Compile the model with Mean Absolute Error loss and Adam optimizer
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

# Add learning rate scheduling
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)

# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Train the model with data augmentation, hyperparameter tuning, and regularization
epochs = 10
batch_size = 5
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
          validation_data=(x_val, y_val), callbacks=[lr_scheduler, early_stopping])

# Evaluate the model on the test set
test_loss = model.evaluate(x_test1, y_test1)

# Make predictions with the model
predictions = model.predict(x_test)
print(predictions)
