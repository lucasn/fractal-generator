import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import os
from itertools import repeat
from collections.abc import Generator
import logging
from numba import jit

BOUND = 2

@jit
def sequence(z: complex, c: complex) -> Generator[complex]:
    """
    Generate an infinite sequence based on the formula :math:`z = z^2 + c`.
    
    This function is commonly used as a basis for generating Mandelbrot and Julia 
    fractal sequences.

    Parameters
    ----------
    z : complex
        The initial value for `z`, typically a complex number.
    c : complex
        A constant complex parameter that remains fixed throughout the sequence.
    
    Yields
    ------
    complex
        The next value in the sequence.
    
    Examples
    --------
    Generate the first few values in a sequence for the Julia set:

    >>> gen = sequence(0.0 + 0.0j, -0.8 + 0.156j)
    >>> [next(gen) for _ in range(5)]
    [0j, (-0.8+0.156j), (-0.18433599999999994-0.09360000000000002j), (-0.7747811991040001+0.1905076992j), (-0.23600727696944535-0.1392035672494403j)]

    Notes
    -----
    - The function does not have a stopping condition and will continue indefinitely.
    - It can be used to determine whether a point belongs to the Mandelbrot or Julia sets 
      based on whether the generated sequence converges or diverges.
    """
    while True:
        yield z
        z = z**2 + c

@jit
def mandelbrot_sequence(c: complex) -> Generator[complex]:
    """
    Generate an infinite sequence for the Mandelbrot set.

    Parameters
    ----------
    c : complex
        A complex number representing a point in the complex plane.

    Returns
    -------
    generator
        A generator that yields the values of the sequence.

    Examples
    --------
    Generate the first few terms in a Mandelbrot sequence:  

    >>> gen = mandelbrot_sequence(-0.7 + 0.3j)
    >>> [next(gen) for _ in range(5)]
    [0j, (-0.7+0.3j), (-0.30000000000000004-0.12j), (-0.6244+0.372j), (-0.44850864-0.16455359999999997j)]
    """ 
    return sequence(0,c)

@jit
def is_in_mandelbrot(c: complex, max_iter: int=100) -> bool:
    """
    Determine whether a complex point is in the Mandelbrot set.

    Parameters
    ----------
    c : complex
        A complex number representing the point in the complex plane to be tested.
    max_iter : int, optional
        The maximum number of iterations to perform. Default is 100.

    Returns
    -------
    bool
        True if the point `c` is within the Mandelbrot set, False otherwise.

    Examples
    --------
    Check if a point is in the Mandelbrot set:

    >>> is_in_mandelbrot(0.0 + 0.0j)
    True

    >>> is_in_mandelbrot(-0.7 + 0.3j)
    False

    >>> is_in_mandelbrot(-1.0 + 0.0j)
    True

    >>> is_in_mandelbrot(0.5 + 0.5j, max_iter=50)
    False
    """
    seq = mandelbrot_sequence(c)
    counter = 0
    z = next(seq)

    while np.abs(z) < BOUND and counter < max_iter:
        z = next(seq)
        counter += 1

    return bool(np.abs(z) < BOUND)


@jit
def _mandelbrot_mask(c_values, max_iter):
    """
    Generate a Mandelbrot set membership mask for a list of complex values.

    Parameters
    ----------
    c_values : list of complex
        A list of complex numbers representing points in the complex plane to test 
        for Mandelbrot set membership.
    max_iter : int
        The maximum number of iterations used to determine if each point in `c_values` 
        is in the Mandelbrot set. Higher values yield a more accurate determination 
        but increase computation time.

    Returns
    -------
    list of bool
        A list of boolean values, where each value corresponds to whether the 
        corresponding complex number in `c_values` is part of the Mandelbrot set.

    Examples
    --------
    >>> c_values = [0+0j, -1+0j, 0.3+0.5j]
    >>> _mandelbrot_mask(c_values, max_iter=100)
    [True, True, True]
    """
    return [is_in_mandelbrot(c, max_iter) for c in c_values]


def _generate_mandelbrot_mask(zmin: complex, zmax: complex, n_points: int, max_iter: int) -> np.ndarray:
    """
    Generate a mask indicating points in the Mandelbrot set within a specified range.

    This function creates a 2D grid of complex numbers, defined by `zmin`, `zmax`, 
    and `n_points`, and evaluates each point to determine whether it belongs to 
    the Mandelbrot set based on the given maximum number of iterations (`max_iter`). 
    The result is a boolean matrix (`True` for points in the Mandelbrot set, `False` 
    otherwise) indicating the membership of each point.

    Parameters
    ----------
    zmin : float
        The minimum value for both the real and imaginary parts of the complex grid.
    zmax : float
        The maximum value for both the real and imaginary parts of the complex grid.
    n_points : int
        The number of points along each axis, creating an `n_points x n_points` grid.
    max_iter : int
        The maximum number of iterations to use in determining if a point is in the Mandelbrot set.

    Returns
    -------
    numpy.ndarray of bool
        A 2D boolean matrix where `True` indicates points in the Mandelbrot set and 
        `False` indicates points outside the set. The shape of the matrix is 
        `(n_points, n_points)`.
    """
    real_range = np.linspace(zmin, zmax, n_points)
    imag_range = np.linspace(zmin, zmax, n_points)
    real_matrix, imag_matrix = np.meshgrid(real_range, imag_range)

    c_matrix = real_matrix + 1j*imag_matrix
    original_shape = c_matrix.shape
    c_values = np.reshape(c_matrix, -1)

    c_values_mask = _mandelbrot_mask(c_values, max_iter)

    c_mask_matrix = np.reshape(c_values_mask, original_shape)

    return c_mask_matrix


def plot_mandelbrot(
        zmin: float=-2, 
        zmax: float=2, 
        pixel_size :float=1e-2, 
        output: str='mandelbrot.png', 
        max_iter: int=100
        ) -> None:
    """
    Generate and save an image of the Mandelbrot set over a specified range.

    Parameters
    ----------
    zmin : float, optional
        The minimum value for both real and imaginary components of the complex plane.
        Default is -2.
    zmax : float, optional
        The maximum value for both real and imaginary components of the complex plane.
        Default is 2.
    pixel_size : float, optional
        The pixel resolution of the image. Smaller values increase the detail level 
        but also increase computation time. Default is 1e-2.
    output : str, optional
        The filename (including path if needed) where the Mandelbrot image will be saved.
        Default is 'mandelbrot.png'.
    max_iter : int, optional
        The maximum number of iterations used to determine if points are in the Mandelbrot set.
        Higher values increase the accuracy of the fractal but may increase computation time.
        Default is 100.

    Returns
    -------
    None
        The function saves the Mandelbrot plot as an image file and does not return any value.
    """
    n_points = int((zmax - zmin)//pixel_size)
    mask = _generate_mandelbrot_mask(zmin, zmax, n_points, max_iter)
    _plot_fractal(output, mask)

@jit
def julia_sequence(z: complex,c :complex) -> Generator[complex]:
    """
    Generate an infinite sequence for the Julia set starting from a given point.

    Parameters
    ----------
    z : complex
        The initial value for `z`, typically a complex number representing 
        the starting point of the sequence.
    c : complex
        A constant complex parameter that remains fixed throughout the sequence.

    Returns
    -------
    generator
        A generator that yields the values of the sequence.

    Examples
    --------
    Generate the first few terms in a Julia sequence:

    >>> gen = julia_sequence(0.0 + 0.0j, -0.8 + 0.156j)
    >>> [next(gen) for _ in range(5)]
    [0j, (-0.8+0.156j), (-0.18433599999999994-0.09360000000000002j), (-0.7747811991040001+0.1905076992j), (-0.23600727696944535-0.1392035672494403j)]

    >>> gen = julia_sequence(1.0 + 1.0j, 0.355 + 0.355j)
    >>> [next(gen) for _ in range(3)]
    [(1+1j), (0.355+2.355j), (-5.0649999999999995+2.02705j)]
    """
    return sequence(z,c)

@jit
def is_in_julia(z: complex, c: complex, max_iter :int=100) -> bool:
    """
    Determine whether a complex point is in the Julia set.

    Parameters
    ----------
    z : complex
        A complex number representing the point in the complex plane to be tested.
    c : complex
        A constant complex parameter that defines the specific Julia set.
    max_iter : int, optional
        The maximum number of iterations to perform. Default is 100.

    Returns
    -------
    bool
        True if the point `z` is within the Julia set for the given `c`, False otherwise.

    Examples
    --------
    Check if a point is in the Julia set:

    >>> is_in_julia(0.0 + 0.0j, -0.8 + 0.156j)
    True

    >>> is_in_julia(0.5 + 0.5j, 0.355 + 0.355j)
    False

    >>> is_in_julia(-1.0 + 0.0j, 0.355 + 0.355j)
    False

    >>> is_in_julia(0.3 + 0.3j, 0.355 + 0.355j, max_iter=50)
    True
    """
    seq = julia_sequence(z, c)
    counter = 0
    z = next(seq)

    while np.abs(z) < BOUND and counter < max_iter:
        z = next(seq)
        counter += 1

    return bool(np.abs(z) < BOUND)


@jit
def _julia_mask(z_values, c, max_iter):
    """
    Generate a Julia set membership mask for a list of complex values.

    Parameters
    ----------
    z_values : list of complex
        A list of complex numbers representing points in the complex plane to test 
        for Julia set membership.
    c : complex
        The complex parameter that defines the specific Julia set. 
    max_iter : int
        The maximum number of iterations used to determine if each point in `z_values` 
        is in the Julia set. Higher values yield a more accurate determination 
        but increase computation time.

    Returns
    -------
    list of bool
        A list of boolean values, where each value corresponds to whether the 
        corresponding complex number in `z_values` is part of the Julia set.

    Examples
    --------
    >>> z_values = [0+0j, 0.3+0.5j, -0.7+0.2j]
    >>> c = -0.8 + 0.156j
    >>> _julia_mask(z_values, c, max_iter=100)
    [True, False, False]
    """
    return [is_in_julia(z, c, max_iter) for z in z_values]


def _generate_julia_mask(c: complex, z_min: float, z_max: float, n_points: int, max_iter: int) -> np.ndarray:
    """
    Generate a mask indicating points in the Julia set for a given constant.

    This function creates a 2D grid of complex points in the complex plane, 
    defined by `z_min`, `z_max`, and `n_points`, and evaluates each point 
    to determine whether it belongs to the Julia set for a given complex 
    constant `c`. The result is a boolean matrix (`True` for points in the 
    Julia set, `False` otherwise) indicating the membership of each point.

    Parameters
    ----------
    c : complex
        The constant complex parameter that defines the specific Julia set.
    z_min : float
        The minimum value for both real and imaginary parts of the complex grid.
    z_max : float
        The maximum value for both real and imaginary parts of the complex grid.
    n_points : int
        The number of points along each axis, creating an `n_points x n_points` grid.
    max_iter : int
        The maximum number of iterations to use in determining if a point is in the Julia set.

    Returns
    -------
    numpy.ndarray of bool
        A 2D boolean matrix where `True` indicates points in the Julia set and 
        `False` indicates points outside the set. The shape of the matrix is 
        `(n_points, n_points)`.
    """
    real_range = np.linspace(z_min, z_max, n_points)
    imag_range = np.linspace(z_min, z_max, n_points)

    real_matrix, imag_matrix = np.meshgrid(real_range, imag_range)
    z_matrix = real_matrix + 1j*imag_matrix
    original_shape = z_matrix.shape
    z_values = np.reshape(z_matrix, -1)

    z_values_mask = _julia_mask(z_values, c, max_iter)

    z_mask_matrix = np.reshape(z_values_mask, original_shape)

    return z_mask_matrix

def plot_julia(c=-0.8+0.156j, zmin=-2, zmax=2, pixel_size=1e-2, output='julia.png', max_iter=100):
    """
    Generate and save an image of the Julia set for a specified complex parameter.

    Parameters
    ----------
    c : complex, optional
        The complex parameter defining the specific Julia set. Default is -0.8 + 0.156j.
    zmin : float, optional
        The minimum value for both the real and imaginary components of the complex plane.
        Default is -2.
    zmax : float, optional
        The maximum value for both the real and imaginary components of the complex plane.
        Default is 2.
    pixel_size : float, optional
        The pixel resolution of the image. Smaller values increase the detail level 
        but also increase computation time. Default is 1e-2.
    output : str, optional
        The filename (including path if needed) where the Julia set image will be saved.
        Default is 'julia.png'.
    max_iter : int, optional
        The maximum number of iterations used to determine if points are in the Julia set.
        Higher values increase the accuracy of the fractal but may increase computation time.
        Default is 100.

    Returns
    -------
    None
        The function saves the Julia set plot as an image file and does not return any value.
    """
    n_points = int((zmax - zmin)//pixel_size)
    values = _generate_julia_mask(c, zmin, zmax, n_points, max_iter)
    _plot_fractal(output, values)


def _plot_fractal(output_filename: str, mask: np.ndarray) -> None:
    """
    Plot and save a fractal image based on a boolean mask.

    Parameters
    ----------
    output_filename : str
        The filename (including path if needed) where the generated fractal image will be saved.
    mask : numpy.ndarray
        A 2D boolean array where `True` values indicate points in the fractal set, 
        and `False` values represent points outside the set.

    Returns
    -------
    None
        This function saves the fractal plot as an image file and does not return any value.
    """

    # Find indices where the matrix is True
    true_indices = np.argwhere(mask)

    # Get the bounds of the True values
    x_min, y_min = true_indices.min(axis=0)
    x_max, y_max = true_indices.max(axis=0)


    plt.imshow(mask , cmap="Greys", interpolation="nearest")
    plt.axis('off')

    # Set limits to fit the True values
    plt.xlim(y_min - 0.5, y_max + 0.5)
    plt.ylim(x_max + 0.5, x_min - 0.5)  # Invert y-axis for correct orientation

    output_filepath = f"{os.getcwd()}/{output_filename}"

    logging.info(f'Output file saved to {output_filepath}')

    plt.savefig(output_filepath, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
