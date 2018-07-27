
print("Imported up_convolution module")

class convolution:

	def __init__(self,input_h,input_w,input_d,kernel_h,kernel_w,kernel_channels,stride_h = 1,stride_w = 1,padding_h = 0,padding_w = 0):
		''' Records values of a convolution. 

		Arguments :
			input_h         -- Int.
                               Input Height.
			input_w         -- Int.
                               Input Width.
			input_d         -- Int.
                               Input Depth.
			kernel_h        -- Int.
						       Kernel Depth
			kernel_w        -- Int.
						       Kernel Width
			kernel_channels -- Int.
                               Kernel Depth
			stride_h        -- Int.
						       Stride in X-axis.
			stride_w        -- Int.
						       Stride in Y-axis.
			padding_h       -- Int.
                               Padding in the X-axis.
			padding_w       -- Int. 
                               Padding in the Y-axis.
		'''

		self.input_h = input_h
		self.input_w = input_w
		self.input_d = input_d

		self.kernel_h = kernel_h
		self.kernel_w = kernel_w
		self.kernel_d = self.input_d
		self.kernel_channels = kernel_channels

		self.stride_h = stride_h
		self.stride_w = stride_w

		self.padding_h = padding_h
		self.padding_w = padding_w

		self.output_h = ((self.input_h  + 2*self.padding_h - self.kernel_h)//(self.stride_h)) + 1
		self.output_w = ((self.input_w  + 2*self.padding_w - self.kernel_w)//(self.stride_w)) + 1
		self.output_d = kernel_channels


class trans_convolve:

	'''
	Description:
	A class that outputs needed information for reversing a convolution OR performing an arbitrary trans_convolution

	'''

	def __init__(self,corres_convolution = None,random_upsample = False,input_h = -1,input_w = -1,input_d = -1,kernel_h = -1, kernel_w = -1, kernel_d = -1,stride_h = -1,stride_w = -1,padding = 'NULL'):

		'''
		Description:
			Constructor

		Arguments:
		corres_convolution : The convolution you would like to reverse. 
		
		random_upsample : If you are not trying to reverse a convolution but instead you want to do some random transposed convolution
		
		input_h : Valid if random_sample == True. X - dimension of input
		input_w : Valid if random_sample == True. Y - dimension of input
		input_d : Valid if random_sample == True. D - dimension of input
		
		kernel_h : Valid if random_sample == True. X - dimension of kernel
		kernel_w : Valid if random_sample == True. Y - dimension of kernel
		kernel_d : Valid if random_sample == True. D - dimension of kernel

		stride_h: Valid if random_sample == True. Stride along the X dimension of the corresponding convolution (hypothetically created).
		stride_w: Valid if random_sample == True. Stride along the Y dimension of the corresponding convolution (hypothetically created).

		padding : Valid if random_sample == True.
		'''

		if random_upsample == False:

			self.corres_convolution = corres_convolution 
			self.padding = 'VALID'

			if self.padding == 'VALID':

				# No padding will be put with the input : 'VALID' padding
				self.input_h = self.corres_convolution.output_h
				self.input_w = self.corres_convolution.output_w
				self.input_d = self.corres_convolution.output_d

				# The output will be the input of the convolution + the padding that was added. Later we can mask this.
				self.output_h = self.corres_convolution.input_h + self.corres_convolution.padding_h
				self.output_w = self.corres_convolution.input_w + self.corres_convolution.padding_w
				self.output_d = self.corres_convolution.input_d

				# Strides in both dimensions
				self.stride_h = self.corres_convolution.stride_h
				self.stride_w = self.corres_convolution.stride_w

		elif random_upsample == True:

			self.input_h = input_h
			self.input_w = input_w
			self.input_d = input_d

			if padding == 'NULL':
				raise ValueEror("Padding not Available.")

			elif padding != 'VALID' and padding != 'NULL':
				raise ValueError("Padding type not known.")

			elif padding == 'VALID':
				self.padding = 'VALID'

				# Calculating The ouput dimension of the transpose convoltuion
				self.output_h =  (input_h - 1)*stride_h + kernel_h
				self.output_w =  (input_w - 1)*stride_w + kernel_w
				self.output_d = kernel_d
			else : 
				self.padding = 'SAME'

				# Calculating The ouput dimension of the transpose convoltuion				
				self.output_h =  (input_h - 1)*stride_h + 1
				self.output_w =  (input_w - 1)*stride_w + 1
				self.output_d = kernel_d