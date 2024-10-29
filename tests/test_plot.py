from os import getcwd, remove
from os.path import exists
from fractal.fractal import plot_mandelbrot, plot_julia

def test_mandelbrot_plot():
    zmin = -2
    zmax = 2
    pixel_size = 1e-2
    max_iter = 100
    output_filename = 'test.png'
    output_filepath = f'{getcwd()}/{output_filename}'

    plot_mandelbrot(zmin=zmin, zmax=zmax, pixel_size=pixel_size, max_iter=max_iter, output=output_filename)

    assert exists(output_filepath) == True

    remove(output_filepath)

def test_julia_plot():
    zmin = -2
    zmax = 2
    pixel_size = 1e-2
    max_iter = 100
    c=-0.8+0.156j
    output_filename = 'test.png'
    output_filepath = f'{getcwd()}/{output_filename}'

    plot_julia(c=c, zmin=zmin, zmax=zmax, pixel_size=pixel_size, output=output_filename, max_iter=max_iter)

    assert exists(output_filepath) == True

    remove(output_filepath)