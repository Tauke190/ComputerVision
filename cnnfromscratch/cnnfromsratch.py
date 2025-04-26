from keras.datasets import mnist
import numpy as np



#this function is used for forward pass 
def image_correlation(input, kernel, mode='valid'):
    # 'valid' ensures no padding (output size = input_size - kernel_size + 1).
    # 28 x 28 --> 26 x 26
    output_shape = [input.shape[0] - kernel.shape[0] + 1, input.shape[1] - kernel.shape[1] + 1]
    output = np.zeros(output_shape)

    #iterating over the input image
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            #extract the 3 x 3 patch of the image at position i,j and aply kernel
            #as shown in fig 3
            output[i, j] = np.sum(input[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return output


def image_convolution(input, kernel, mode='full'):
    #this flips the kernel left/right followed by up/down
    kernel = np.flipud(np.fliplr(kernel))
    
    # padding ensures the output size is input_size + kernel_size - 1
    # retains the original size
    if mode == 'full':
    # Adds kernel_height - 1 pixels top/bottom.
    # Adds kernel_width - 1 pixels left/right.
        padded_input = np.pad(input, [(kernel.shape[0]-1, kernel.shape[0]-1), (kernel.shape[1]-1, kernel.shape[1]-1)], mode='constant')
    else:
        padded_input = input
    output_shape = [padded_input.shape[0] - kernel.shape[0] + 1, padded_input.shape[1] - kernel.shape[1] + 1]
    # initializing the output array.
    output = np.zeros(output_shape)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # convolve
            output[i, j] = np.sum(padded_input[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return output


class convolution:
    def __init__(self,input_shape,kernel_size,depth):
        input_depth, input_height, input_width = input_shape
        #defining the attributes for forward and backward methods

        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1 , input_width - kernel_size + 1 )
        self.kernels_shape = (depth,input_depth,kernel_size,kernel_size) 

        self.kernels = np.random.randn(*self.kernels_shape) # unpacking operator
        self.biases = np.random.randn(*self.output_shape)  # each convolve operation has one bias


    def forward(self,input):
        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += image_correlation(self.input[j],self.kernels[i,j],'valid') # input matrix and kernel matrix , input_depth is 1
        return self.output
    
    def backward(self,output_gradient, learning_rate):

        kernels_gradient = np.zeros(self.kernels_shape); # kernel_shap = 3 x 1 x 3 x 3
        input_gradient = np.zeros(self.input_shape) # input_shape = 28 x 28 x 3


        for i in range(self.depth):
            for j in range(self.input_depth):
                 kernels_gradient[i, j] = image_correlation(self.input[j],output_gradient[i],"valid")
                 #computing the gradients with respect to input
                 input_gradient[j] += image_convolution(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient # dL/dB --> How loss changes with respect to bias

        return input_gradient



class sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, output_gradient):
        # The derivative of the sigmoid function is sigmoid(x) * (1 - sigmoid(x))
        return output_gradient * self.output * (1 - self.output) #dL/dy
    

class average_pooling:
    def __init__(self,input_shape = (26,26,2), pool_size = 2 ,stride = 2):
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.stride = stride
        self.output_shape =(
            self.input_shape[0],
            (self.input_shape[1] - self.pool_size) // 2 + 1,
            (self.input_shape[2] - self.pool_size) // 2 + 1
        )

    

    def forward(self,input):
        # Storing the input for potential future use
        self.input = input

        # Extracting the dimensions of the input
        depth, input_height, input_width = self.input_shape

        # Initializing an array for the output with the calculated output shape
        output = np.zeros(self.output_shape)


        for d in range(depth):
            for i in range(0,input_height,self.stride):
                for j in range(0, input_width, self.stride):
                    # Defining the current window fur pooling
                    h_start, h_end = i, i + self.pool_size
                    w_start, w_end = j, j + self.pool_size

                    window = input[d, h_start:h_end, w_start:w_end]

                    # Calculating the average value of the current window
                    # and storing it in the corresponding position in the output
                    output[d, i // self.stride, j // self.stride] = np.mean(window)

        return output

    def backward(self,output_gradient):
        depth, input_height, input_width = self.input_shape
        input_gradient = np.zeros(self.input_shape)

        for d in range(depth):
            for i in range(0, input_height, self.stride):
                for j in range(0, input_width, self.stride):
                    h_start, h_end = i, i + self.pool_size
                    w_start, w_end = j, j + self.pool_size

                    # Distribute the gradient equally to each element in the window
                    input_gradient[d, h_start:h_end, w_start:w_end] += \
                        output_gradient[d, i // self.stride, j // self.stride] / (self.pool_size * self.pool_size)
                    
        return input_gradient


class flatten:
    def forward(self , input):
        self.input_shape = input.shape
        return input.flatten()
    
    def backward(self , output_gradient):
        return output_gradient.reshape(self.input_shape)
    

class dense_layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size)


    def forward(self, input):
        self.input = input
      #  print(self.input.shape)
        eval = np.dot(self.weights, input) + self.biases
      #  print(eval.shape) = 10,
        return eval

    def backward(self , output_gradient, learning_rate):

        # Gradient of loss w.r.t. weights
        weights_gradient = np.dot(output_gradient.reshape(-1,1) , self.input.reshape(1,-1))

        # Gradient of loss w.r.t. biases
        biases_gradient = output_gradient

        # Gradient computation of loss w.r.t. input
        input_gradient = np.dot(self.weights.T,output_gradient)

        # Updating weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_gradient    
    

class softmax:
    def forward(self,input):
        exps = np.exp(input - np.max(input, axis=-1, keepdims=True))
        self.output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, dL_dZ):
        return dL_dZ
    
    
class cross_entropy_loss:
    def __init__(self):
        pass

    def compute_loss(self, t_list , p_list):
        #Ensure inputs are 2D arrays
        t_list = np.atleast_2d(np.float_(t_list))
        p_list = np.atleast_2d(np.float_(p_list))
        # compute cross entropy loss
        losses = -np.sum(t_list * np.log(p_list + 1e-15),axis =1)
        return np.mean(losses)
    
    def compute_dloss(self,t_list,p_list):
        return p_list - t_list

#one hot code for all the labels
def one_hot_encode(y):
    one_hot = np.zeros((y.size, y.max()+1))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


#batch,num_imgs,w,h and normalize
train_images = train_images.reshape((-1, 1, 28, 28)) / 255.0  
test_images = test_images.reshape((-1, 1, 28, 28)) / 255.0

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)


#Intialize layers

cnn_layer = convolution(input_shape=(1,28,28),kernel_size=3,depth=2) # depth = no of kernels
sigmoid_layer = sigmoid()
avg_pooling_layer = average_pooling(input_shape=(2,26,26),pool_size=2,stride=2)
flatten_layer = flatten()
denselayer = dense_layer(input_size=(13*13*2),output_size=10)
softmax_layer = softmax()
cross_entropy_layer = cross_entropy_loss()


#hyperparameters
num_epochs = 10
learning_rate = 0.01

# Store results over time
train_accuracies = []
train_losses = []

train_images = train_images[:1000]

for epoch in range(num_epochs):
    correct_train_predictions = 0
    total_loss = 0

    #loop over all the images
    for i in range(len(train_images)):
        image = train_images[i]
        label = train_labels[i]

        cnn_output = cnn_layer.forward(image)
        sigmoid_output = sigmoid_layer.forward(cnn_output)
        avg_pooling_output = avg_pooling_layer.forward(sigmoid_output)
        flatten_output = flatten_layer.forward(avg_pooling_output)
        dense_output = denselayer.forward(flatten_output)
        predictions = softmax_layer.forward(dense_output)

        loss = cross_entropy_layer.compute_loss(label,predictions)
        total_loss += loss

        if(np.argmax(predictions) == np.argmax(label)):
            correct_train_predictions += 1
        # Back pass
        grad_back = cross_entropy_layer.compute_dloss(label,predictions)

        grad_back = softmax_layer.backward(grad_back)


        grad_back = denselayer.backward(grad_back,learning_rate)

        # print("Gradient norm:", np.linalg.norm(grad_back))


      
        grad_back = flatten_layer.backward(grad_back)
       
        grad_back = avg_pooling_layer.backward(grad_back)
    
        grad_back = sigmoid_layer.backward(grad_back)
        grad_back = cnn_layer.backward(grad_back,learning_rate)


    train_accuracy = correct_train_predictions/len(train_images)
    average_loss = total_loss/len(train_images)

    train_accuracies.append(train_accuracy)
    train_losses.append(average_loss)

    # Print summary after each epoch
    print(f"Epoch {epoch + 1}, Training Accuracy: {train_accuracy * 100:.2f}%, Average Loss: {average_loss:.4f}")    














