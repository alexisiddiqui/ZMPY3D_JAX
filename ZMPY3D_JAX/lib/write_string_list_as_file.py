# This function is a basic file I/O operation and does not involve numerical computations.
# It would remain largely unchanged in a JAX conversion, as JAX focuses on numerical operations and not file system interactions.


from typing import Sequence


def write_string_list_as_file(string_list: Sequence[str], filename: str) -> None:
    """Writes a list of strings to a specified file, with each string on a new line.

    Args:
        string_list (list): A list of strings to be written to the file.
        filename (str): The path to the output file.
    """
    with open(filename, "w") as file:
        for line in string_list:
            file.write(f"{line}\n")
