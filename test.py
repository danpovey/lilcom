import lilcom
import numpy
import csv

# read CSV file & load into list
with open("sample.txt", 'r') as my_file:
    reader = csv.reader(my_file)
    data = list(reader)


data = [int(d) for d in data[0]]

print("Data Loaded from sample.txt")

## Making a numpy array from the given data
inputArray = numpy.array(data, dtype=numpy.int16)

## Compression
# Making an empty numpy array for output
compressed = numpy.zeros(inputArray.shape, dtype = numpy.int8)
lilcom.compress(inputArray, compressed)

## Decompression (retrieval)
# Making an empty numpy array for retieved signal
retrieved = numpy.zeros(inputArray.shape, dtype = numpy.int16)
lilcom.decompress(compressed, retrieved)

for i in range(len(data)):
    print("The original index: ", inputArray[i], " compressed is: ", compressed[i], " retrieved: ", retrieved[i])