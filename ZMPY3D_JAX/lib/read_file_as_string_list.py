# This function is a basic file I/O operation and does not involve numerical computations.
# It would remain largely unchanged in a JAX conversion.

from typing import List


def read_file_as_string_list(filename: str) -> List[str]:
    """Reads a file and returns its content as a list of strings,
    with each line as an element and stripped of leading/trailing whitespace.

    Args:
        filename (str): The path to the input file.

    Returns:
        list: A list of strings, where each string is a line from the file.
    """
    with open(filename, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]
