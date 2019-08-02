import lilcomlib
import numpy

# Sample test array to test the function.
numpy_array = numpy.array([[1]* 3]*100)

# Running the function in lilcommodule.c
val1 = lilcomlib.compress(numpy_array)

# numpy_array = numpy.array([[5]*10] * 100)
# val2 = lilcomlib.decompress(numpy_array)


print (val1)
#print (val2)