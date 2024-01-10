# slim-flatboi
Script to reflatten the scroll segments used in the Vesuvius Challenge ( https://scrollprize.org/ ) using Scalable Locally Injective Mappings (SLIM) as described in the [SLIM paper](https://igl.ethz.ch/projects/slim/SLIM2017.pdf).

## Installation
### Prerequisites
- Conda (Anaconda or Miniconda).
- Flattened segment(s) with `.obj`, `.tif` and `.mtl` in the same folder.
### Setting up the Environment
1. Clone or download this repository to your local machine.
2. Create a Conda environment with the required dependencies:
```console
conda env create -f environment.yml
```
3. Activate the environment:
```console
conda activate flatboi-env
```
## Usage
Run the script from the command line, specifying the path to your `.obj` file and the number of iterations as arguments:
```console
python flatboi.py <path_to_obj_file> <number_of_iterations>
```
Example:
```console
python flatboi.py /path/to/segment.obj 20
```

## Advice
Select a number of iterations that makes the procedure converge. I would put at least `20`.

## Output
The script will generate modified `_flatboi.obj`, `_flatboi.png`, and `_flatboi.mtl` files in the same directory as the input file. Additionally, it will create a `energies_flatboi.txt` file containing Symmetric Dirichlet Energy values computed during the iterative process.

## License
See the LICENSE file.

## Author
Dr. Giorgio Angelotti