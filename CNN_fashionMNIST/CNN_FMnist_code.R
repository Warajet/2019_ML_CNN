#Install keras package
install.packages("keras")
library(keras)

#Import the fashionMNIST dataset.
fashion_mnist <- dataset_fashion_mnist()
str(fashion_mnist)

#Set the train & test samples
c(fmnist_train_image,fmnist_train_label)  %<-%fashion_mnist$train # 60,000 train samples
c(fmnist_test_image, fmnist_test_label) %<-% fashion_mnist$test  # 10,000 test samples

class_labels <- c('T-shirt/top','Trouser','Pullover','Dress','Coat', 'Sandal',
                  'Shirt','Sneaker','Bag','Ankle boot')

output_n <- length(class_labels)
dim(fmnist_train_image)
dim(fmnist_test_image)

rescale_image <- function(x)
{
  x <- x / 255
}


#Normalize the image from 0-255 grayscale to 0-1 grayscale
fmnist_train_image <- rescale_image( x = fmnist_train_image)
fmnist_test_image <- rescale_image( x = fmnist_test_image)

img_width <- dim(fmnist_train_image)[1]
img_height <- dim(fmnist_train_image)[2]
channels <- 1

#Initialise the CNN model
cnn_model <- keras_model_sequential()

cnn_model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = 'same', input_shape = c(img_width, img_height, channels)) %>%
  layer_activation('relu') %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = 'same') %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation('relu') %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation('softmax')

# compile
cnn_model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
  
cnn_model %>% fit(fmnist_train_image, fmnist_train_label, epochs = 5)

score <- cnn_model %>% evaluate(fmnist_test_image, fmnist_test_label)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")









