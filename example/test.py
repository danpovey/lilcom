import spam
print("Running the command \"ls -l\":")
spam.system("ls -l")
print("Try running a wrong command to test the error handling:")
spam.system("wrongcommand")

