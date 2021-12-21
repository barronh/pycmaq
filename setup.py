import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("pycmaq/__init__.py", "r") as fh:
    for line in fh:
        if line.startswith('__version__'):
            exec(line)
            break
    else:
        __version__ = 'x.y.z'

setuptools.setup(
    name="pycmaq",
    version=__version__,
    author="Barron H. Henderson",
    author_email="barronh@gmail.com",
    description="Utilities for working with CMAQ data in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/barronh/pycmaq",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["numpy", "matplotlib", "xarray", "pyproj", "netCDF4"],
    include_package_data=True,
    extras_require={
        "shapely":  ["shapely"],
    }
)
