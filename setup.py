import setuptools
import cr5

setuptools.setup(
    name="cr5-lib",
    version=cr5.__version__,
    author="Wanhao Zhou",
    author_email="wanhao.zhou@gmail.com",
    description="A memory efficient implementation of Cr5 library ",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
