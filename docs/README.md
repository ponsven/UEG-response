# Building the documentation

To build the documentation locally, run the following comands in the base directory of the project, i.e. `..` from here:

* [Optional] Activate virtual enviroment.
* Install *UEG-response* in this enviroment.
* Install Sphinx and and extensions:


```
    pip install sphinx sphinx-rtd-theme
    pip install furo
```

* Build HTML documentation:

```
    sphinx-build -b html docs/source/ docs/build/html
```

* View the documentation by opening the file `docs/build/html/index.html` in a browser.
