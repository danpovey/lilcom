# lilcom
Small compression utility


Note: you can run
```
python setup.py build
python setup.py install
```

and then from python, do:

```
>>> import lilcom
>>> lilcom.lilcom("a")
```
and it will segfault.  Currently the interface in lilcommodule.c is just a
skeleton I copied from somewhere, it needs to be actually implemented.
