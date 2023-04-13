// C ML
// By JJ Reibel

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void train_val_test_split(double *X, double *y, int n_samples, double val_size, double test_size, int epochs, int random_state, double **X_train_epoch, double **X_val_epoch, double **X_test_epoch, double **y_train_epoch, double **y_val_epoch, double **y_test_epoch) {
    // Set the random seed if provided
    if (random_state != 0) {
        srand(random_state);
    }
    // Create a list of indices that correspond to the samples in the dataset
    int *idx = (int*)malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        idx[i] = i;
    }
    // Shuffle the indices
    for (int i = 0; i < n_samples; i++) {
        int j = rand() % n_samples;
        int temp = idx[i];
        idx[i] = idx[j];
        idx[j] = temp;
    }
    // Calculate the number of samples to allocate to the validation and test sets
    int n_val = ceil(n_samples * val_size);
    int n_test = ceil(n_samples * test_size);
    // Initialize the starting and ending indices of each epoch
    int *epoch_start_idx = (int*)malloc(epochs * sizeof(int));
    for (int i = 0; i < epochs; i++) {
        epoch_start_idx[i] = i * n_samples / epochs;
    }
    int *epoch_end_idx = (int*)malloc(epochs * sizeof(int));
    for (int i = 0; i < epochs - 1; i++) {
        epoch_end_idx[i] = epoch_start_idx[i + 1];
    }
    epoch_end_idx[epochs - 1] = n_samples;
    // Initialize the lists to hold the indices of the samples in each set for each epoch
    int **train_idx_epoch = (int**)malloc(epochs * sizeof(int*));
    int **val_idx_epoch = (int**)malloc(epochs * sizeof(int*));
    int **test_idx_epoch = (int**)malloc(epochs * sizeof(int*));
    for (int i = 0; i < epochs; i++) {
        train_idx_epoch[i] = (int*)malloc((n_samples - n_val - n_test) * sizeof(int));
        val_idx_epoch[i] = (int*)malloc(n_val * sizeof(int));
        test_idx_epoch[i] = (int*)malloc(n_test * sizeof(int));
    }
    // Loop through each epoch
    for (int i = 0; i < epochs; i++) {
        // Get the indices of the samples in the current epoch
        int *epoch_indices = &idx[epoch_start_idx[i]];
        // Calculate the indices of the samples to allocate to the validation and test sets
        int *val_idx = &epoch_indices[0];
        int *test_idx = &epoch_indices[n_val];
        int *train_idx = &epoch_indices[n_val + n_test];
        // Add the indices to the appropriate lists for the current epoch
        for (int j = 0; j < (n_samples - n_val - n_test); j++) {
            train_idx_epoch[i][j] = train_idx[j];
        }
        for (int j = 0; j < n_val; j++) {
            val_idx_epoch[i][j] = val_idx[j];
        }
        for (int j = 0; j < n_test; j++) {
            test_idx_epoch[i][j] = test_idx[j];
        }
    }
    //

// Initialize lists to hold the data for each epoch
double **X_train_epoch = malloc(epochs * sizeof(double *));
double **X_val_epoch = malloc(epochs * sizeof(double *));
double **X_test_epoch = malloc(epochs * sizeof(double *));
double **y_train_epoch = malloc(epochs * sizeof(double *));
double **y_val_epoch = malloc(epochs * sizeof(double *));
double **y_test_epoch = malloc(epochs * sizeof(double *));

// Loop through each epoch
for (int i = 0; i < epochs; i++) {
    // Get the indices of the samples for the current epoch
    int *train_idx = &train_idx_epoch[i][0];
    int *val_idx = &val_idx_epoch[i][0];
    int *test_idx = &test_idx_epoch[i][0];
    // Get the data for the current epoch
    double *X_train = malloc((n_samples - n_val - n_test) * sizeof(double));
    double *X_val = malloc(n_val * sizeof(double));
    double *X_test = malloc(n_test * sizeof(double));
    double *y_train = malloc((n_samples - n_val - n_test) * sizeof(double));
    double *y_val = malloc(n_val * sizeof(double));
    double *y_test = malloc(n_test * sizeof(double));
    for (int j = 0; j < (n_samples - n_val - n_test); j++) {
        X_train[j] = X[train_idx[j]];
        y_train[j] = y[train_idx[j]];
    }
    for (int j = 0; j < n_val; j++) {
        X_val[j] = X[val_idx[j]];
        y_val[j] = y[val_idx[j]];
    }
    for (int j = 0; j < n_test; j++) {
        X_test[j] = X[test_idx[j]];
        y_test[j] = y[test_idx[j]];
    }
    // Append the data to the appropriate lists for the current epoch
    X_train_epoch[i] = X_train;
    X_val_epoch[i] = X_val;
    X_test_epoch[i] = X_test;
    y_train_epoch[i] = y_train;
    y_val_epoch[i] = y_val;
    y_test_epoch[i] = y_test;
}

/**
 * Trains the model using the given training data and labels for the specified number of epochs.
 * @param X_train - the training data, a 2D array of shape (n_samples, n_features)
 * @param y_train - the training labels, a 1D array of shape (n_samples,)
 * @param n_samples - the total number of samples in the training set
 * @param n_features - the number of features in the training data
 * @param epochs - the number of times to iterate over the entire training set
 * @param batch_size - the number of samples per batch
 */
void fit(double **X_train, double *y_train, int n_samples, int n_features, int epochs, int batch_size) {
    // Allocate memory for the model's parameters
    double *weights = malloc(n_features * sizeof(double));
    double bias = 0.0;
    
    // Loop over the specified number of epochs
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Shuffle the training data and labels
        shuffle(X_train, y_train, n_samples, n_features);
        
        // Loop over each batch of training data
        for (int batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
            int batch_end = batch_start + batch_size;
            if (batch_end > n_samples) {
                batch_end = n_samples;
            }
            
            // Calculate the gradient of the loss function with respect to the weights and bias
            double *grad_weights = malloc(n_features * sizeof(double));
            double grad_bias = 0.0;
            for (int i = batch_start; i < batch_end; i++) {
                double prediction = predict(X_train[i], weights, bias, n_features);
                double error = prediction - y_train[i];
                for (int j = 0; j < n_features; j++) {
                    grad_weights[j] += error * X_train[i][j];
                }
                grad_bias += error;
            }
            
            // Update the weights and bias using the calculated gradient
            for (int i = 0; i < n_features; i++) {
                weights[i] -= learning_rate * grad_weights[i] / (batch_end - batch_start);
            }
            bias -= learning_rate * grad_bias / (batch_end - batch_start);
            
            free(grad_weights);
        }
    }
    
    // Save the trained model's parameters as instance variables
    this->weights = weights;
    this->bias = bias;
}

/**
 * Evaluates the model on the given test data and labels, and returns the mean squared error.
 * @param X_test - the test data, a 2D array of shape (n_samples, n_features)
 * @param y_test - the test labels, a 1D array of shape (n_samples,)
 * @param n_samples - the total number of samples in the test set
 * @return - the mean squared error of the model's predictions on the test set
 */
double evaluate(double **X_test, double *y_test, int n_samples) {
    double mse = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double prediction = predict(X_test[i], this->weights, this->bias, this->n_features);
        double error = prediction - y_test[i];
        mse += error * error;
    }
    mse /= n_samples;
    return mse;
}


void shuffle(int n, int* arr) {
    // Shuffle the array using the Fisher-Yates algorithm
    srand(time(NULL));
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

double* predict(double* X, int n_samples, double* weights, double bias) {
    // Allocate memory for the predictions
    double* y_pred = malloc(n_samples * sizeof(double));
    // Make predictions using the linear regression model
    for (int i = 0; i < n_samples; i++) {
        double dot_product = 0;
        for (int j = 0; j < n_features; j++) {
            dot_product += weights[j] * X[i*n_features + j];
        }
        y_pred[i] = dot_product + bias;
    }
    return y_pred;
}

// Define the number of features
int n_features = 10;

/* If loading data from CSV file instead:
// Load data from CSV file
FILE *fp = fopen("data.csv", "r");
// Count the number of columns (i.e. features)
int n_features = 0;
char c;
while ((c = fgetc(fp)) != EOF) {
    if (c == ',') {
        n_features++;
    } else if (c == '\n') {
        // We've reached the end of the header row
        break;
    }
}
fclose(fp);

*/


// Example

int main() {
    // Define the input data and labels
    double X[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double y[] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    int n_samples = 10;
    int n_features = 1;
    int n_classes = 2;

    // Split the data into training, validation, and test sets
    int n_val = 2;
    int n_test = 2;
    double **X_train, **X_val, **X_test;
    double *y_train, *y_val, *y_test;
    train_val_test_split(X, y, n_samples, n_features, n_classes, n_val, n_test,
                         &X_train, &X_val, &X_test, &y_train, &y_val, &y_test);

    // Print the shapes of the training, validation, and test sets
    printf("X_train shape: (%d, %d)\n", n_samples - n_val - n_test, n_features);
    printf("y_train shape: (%d,)\n", n_samples - n_val - n_test);
    printf("X_val shape: (%d, %d)\n", n_val, n_features);
    printf("y_val shape: (%d,)\n", n_val);
    printf("X_test shape: (%d, %d)\n", n_test, n_features);
    printf("y_test shape: (%d,)\n", n_test);

    // Free the memory allocated for the training, validation, and test sets
    for (int i = 0; i < n_samples - n_val - n_test; i++) {
        free(X_train[i]);
    }
    free(X_train);
    free(y_train);
    for (int i = 0; i < n_val; i++) {
        free(X_val[i]);
    }
    free(X_val);
    free(y_val);
    for (int i = 0; i < n_test; i++) {
        free(X_test[i]);
    }
    free(X_test);
    free(y_test);

    return 0;
}

