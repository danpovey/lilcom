 #!/bin/bash
echo Checking the packages to be installed

pip3 show numpy
if (($? != 0)); then
    echo Numpy not found. installing numpy...
    pip3 install numpy
fi

pip3 show scipy
if (($? != 0)); then
    echo Numpy not found. installing numpy...
    pip3 install numpy
fi

pip3 show librosa
if (($? != 0)); then
    echo Numpy not found. installing numpy...
    pip3 install numpy
fi

pip3 show pydub
if (($? != 0)); then
    echo Numpy not found. installing numpy...
    pip3 install numpy
fi

pip3 show lilcom
if (($? != 0)); then
    echo Numpy not found. installing numpy...
    python3 ../setup.py install
fi

python3 ./test_reconstruction.py