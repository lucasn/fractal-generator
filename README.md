# Fractal Generator

A simple Python library for generating and visualizing Mandelbrot and Julia fractals. With adjustable settings for resolution, iteration depth, and complex plane boundaries, this library enables you to create high-quality images of intricate fractal patterns with ease.

## Key Features
- Generate Mandelbrot & Julia Sets: Easily create and save fractal images.
- Customizable Detail: Control pixel size, iteration limits, and complex plane range.
- High-Resolution Output: Save fractal visuals in high quality.

## Quick Start

> [!WARNING]
> Python 3.12.2 required

Install the library:
```bash
$ pip install .
```

Command-Line usage:
```bash
$ MandelbrotPlot --zmin=-0.7440+0.1305j\
                --zmax=-0.7425+0.1320j \
                --pixel_size=5e-7\
                --max-iter=50\
                -o "Mandelbrot_tentacle_lowiter.png" 
```
```bash
$ JuliaPlot -c=-0.8j\
            --pixel_size=1e-3\
            --max-iter=50\
            -o "thunder-julia.png" 
```

Python library usage:
```py
from fractal import fractal

zmin = -2
zmax = 2
pixel_size = 1e-2
max_iter = 100
c=-0.8+0.156j

fractal.plot_mandelbrot(
    zmin=zmin, 
    zmax=zmax, 
    pixel_size=pixel_size, 
    max_iter=max_iter
    )
fractal.plot_julia(
    c=c, 
    zmin=zmin, 
    zmax=zmax, 
    pixel_size=pixel_size, 
    max_iter=max_iter
    )

``` 
Open the built-in documentation:
```bash
$ chmod +c ./run_docs.sh
$ ./run_docs.sh
```
The docs will be available at [http://localhost:8000](http://localhost:8000)

## Development
To run automated tests:
```
$ pytest
```