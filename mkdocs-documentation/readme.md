# Generate the NLSE Documentation

We are using the `mkdocs` package to generate the documentation for the `NLSE` package. The documentation is written in markdown and can be found in the `docs` directory.

## Installation

The documentation is based on the `mkdocs` package.

MkDocs requires a recent version of Python and the Python package manager, pip, to be installed on your system.

You can check if you already have these installed from the command line:

```
$ python --version
Python 3.8.2
$ pip --version
pip 20.0.2 from /usr/local/lib/python3.8/site-packages/pip (python 3.8)
```

If you already have those packages installed, you may skip down to Installing MkDocs. Otherwise refer to the appropriate documentation for your system.

### Install the mkdocs package

Install the mkdocs package using pip:

```
pip install mkdocs
```

You should now have the mkdocs command installed on your system. Run `mkdocs
--version` to check that everything worked okay.

```
$ mkdocs --version
mkdocs, version 1.2.0 from /usr/local/lib/python3.8/site-packages/mkdocs (Python 3.8)
```

### Extra packages


You need to install packages to support the markdown extensions : [mkdocsmateral](https://github.com/squidfunk/mkdocs-material), [mkdocstrings](https://pypi.org/project/mkdocstrings/) and [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) used in the documentation. You can install these using pip:

```
pip install mkdocs-material
pip install mkdocs-jupyter
pip install mkdocstrings
```

And associated dependencies:

```
pip install pyqtwebengine
```

### Note

If you are using Windows, some of the above commands may not work out-of-the-box.

A quick solution may be to preface every Python command with `python -m` like this:

```
python -m pip install mkdocs
python -m mkdocs
```

## Testing the site

To test the site locally, run the following command:

```
mkdocs serve
```

Then go to Serving on http://127.0.0.1:8000/ in your browser.

## Building the site

That's looking good. You're ready to deploy the first pass of your MkLorum documentation. First build the documentation:

```
mkdocs build
```

# This will create a new directory, named `site`.

# Documentation of NLSE

## Installation

pip install mkdocs

## Dev server

To launch the dev server : mkdocs serve
Then go to [Here](http://127.0.0.1:8000/)

