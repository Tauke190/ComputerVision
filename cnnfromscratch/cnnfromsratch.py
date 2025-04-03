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


class convultion:
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
                 kernels_gradient[i, j] = image_correlation(self.input[j],output_gradient[i],'valid')
                 #computing the gradients with respect to input
                 input_gradient[j] += image_convolution(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient # dL/dB --> How loss changes with respect to bias

        return input_gradient



class sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))

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

    