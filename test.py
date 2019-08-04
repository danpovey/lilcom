import lilcom
import numpy
import csv

# read CSV file & load into list
with open("sample.txt", 'r') as my_file:
    reader = csv.reader(my_file)
    data = list(reader)


data = [int(d) for d in data[0]]
print(data)

# Sample test array to test the function.
numpy_array = numpy.array(data)
numpy_array = numpy_array.astype(numpy.int16)
# Running the function in lilcommodule.c
val1 = lilcom.compress(numpy_array)

# numpy_array = numpy.array([[5]*10] * 100)
val2 = lilcom.decompress(val1)


print (val1)
print (val2)