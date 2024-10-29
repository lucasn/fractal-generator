import argparse
from time import time
from fractal.fractal import plot_mandelbrot, plot_julia
from copy import deepcopy
import logging


def main():
    logging.basicConfig(
        format='%(asctime)s %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p',
        handlers=[logging.StreamHandler()],
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str, help='Output path/name for the plot')
    parser.add_argument('--zmin', type=float, help='Minimum value for the grid of values to search')
    parser.add_argument('--zmax', type=float, help='Maximum value for the grid of values to search')
    parser.add_argument('--pixel-size', type=float, help='Distance between two points in the grid of search')
    parser.add_argument('--max-iter', type=int, help="Maximum of iterations in the divergence calculation")

    if parser.prog == 'MandelbrotPlot':
        parser.description = "Plot Mandelbrot's fractal"
    else:
        parser.description = "Plot Julia's fractal"
        parser.add_argument('-c', type=complex, help="Complex value C used to generate Julia's sequence")


    args = parser.parse_args()
    params = deepcopy(vars(args))
    for param in vars(args):
        if params[param] is None:
            del params[param]

    start = time()

    if parser.prog == 'MandelbrotPlot':    
        plot_mandelbrot(**params)
    else:
        plot_julia(**params)

    logging.info(f'Execution Time: {time() - start} s')

if __name__=="__main__":
    main()