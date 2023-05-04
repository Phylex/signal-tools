from pathlib import Path
import sys
import csv
import yaml
import click
import re
import numpy as np
import matplotlib.pyplot as plt
from .stream_utils import arrays_to_data_line
from signal_tools.filter import trapezoid_filter


@click.group("signal-io")
@click.argument("file-path", type=click.Path(dir_okay=False))
@click.argument("direction", type=click.Choice(["in", "out", "append"],
                                               case_sensitive=False))
@click.argument("encoding", type=click.Choice(["bin", "utf-8"],
                                              case_sensitive=False))
@click.pass_context
def file_io(ctx, file_path: click.Path, direction: str, encoding: str) -> None:
    """
    Read from and write to files

    FILE-PATH determins the file that is to be read from.
    DIRECTION determins if data should be read from or written to the file and
    ENCODING specifies if the file should be read as binary or utf-8 encoded file
    """
    ctx.obj = {}
    io_file = Path(str(file_path))
    if (direction == 'in' or direction == "append") and not io_file.exists():
        click.echo("File does not exist")
        sys.exit(1)
    ctx.obj = {}
    file_mode = ""
    match encoding:
        case "bin":
            file_mode += "b"
        case _:
            pass
    match direction:
        case "in":
            file_mode += "r"
            in_file = open(io_file, file_mode)
            out_file = click.get_text_stream('stdout')
        case "out":
            file_mode = "w+"
            out_file = open(io_file, file_mode)
            in_file = click.get_text_stream('stdin')
        case _:
            click.echo("Invalid application state")
            sys.exit(2)
    ctx.obj = {'in': in_file,
               'out': out_file}


@click.command()
@click.argument("delimiter", type=str)
@click.option("-c", "--signal-column", type=int, multiple=True, required=True,
              help="index of the column that should be read in")
@click.option("-t", "--column-type", multiple=True,
              type=(int, click.Choice(['int', 'float'])),
              help="Designate the data type of the column "
                   "(either 'int' or 'float'), defaults to float",
              default=[[-1, 'float']])
@click.pass_context
def read_csv(ctx, delimiter: str,
             signal_column: tuple[int],
             column_type: tuple[tuple[int, str]]):
    """
    read in a file of CSV format.
    """
    csvr = csv.reader(
        ctx.obj['in'], delimiter=delimiter, skipinitialspace=True)
    out = ctx.obj['out']
    col_names = next(csvr)
    col_idx = list(range(len(col_names)))
    columns_with_types = list(map(lambda c: c[0], column_type))
    if any(map(lambda i: i not in col_idx and i != -1, columns_with_types)):
        raise ValueError(f"Invalid column index in column format "
                         f"specification. {len(col_names)} available")
    if any(map(lambda i: i not in col_idx, signal_column)):
        raise ValueError(f"Invalid column index in column selection. {len(col_names)} available")

    # generate the description dict for the requested data
    column_descriptions = []
    for i, name in enumerate(col_names):
        if i in signal_column:
            column_descriptions.append(
                {'name': str(name),
                 'type': str(column_type[i][1])
                    if i in columns_with_types else 'float',
                 'shape': [1]
                 })

    # write the meta data for the requested columns to the data stream
    out.write("Metadata:\n")
    out.write(yaml.dump(column_descriptions))
    out.write("\nData:\n")

    # now write the stream info to the
    for line in csvr:
        # filter out the columns that we wanted
        req_data = list(
                map(lambda elem: np.array([elem[1]]),
                    filter(lambda elem: elem[0] in signal_column,
                           enumerate(line))))
        out.write(arrays_to_data_line(req_data)+'\n')


@click.command()
@click.argument("lsb-magnitude", type=float)
@click.option("-c", "--column", type=int, multiple=True, required=True,
              help="Select the column that is to be processed")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Specify a file to write the output of the command to."
              "If not specified, 'stdout' will be used")
@click.pass_context
def digitize(ctx: click.Context, lsb_magnitude: float,
             output: click.Path) -> None:
    if output is not None:
        output_path = Path(str(output))
        out = open(output_path, 'w+')
    else:
        out = click.get_text_stream('stdout')
    y = ctx.obj["y"]
    try:
        x = ctx.obj["x"]
    except KeyError:
        x = ("Measurement Samples", np.arange(len(y[1])))
    measurements = y[1]
    digitized_meas = [m // lsb_magnitude for m in measurements]
    for x_el, y_el in zip(x[1], digitized_meas):
        out.write(f"{x_el},{y_el}\n")
    out.close()


@click.command()
@click.argument("k", type=int)
@click.argument("l", type=int)
@click.argument("m", type=int)
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Specify a file to write the output of the command to. "
              "If not specified, 'stdout' will be used")
@click.pass_context
def apply_trapezoidal_filter(ctx: click.Context, k: int,
                             l: int, m: int,
                             output: click.Path):
    signal = ctx.obj["y"][1]
    try:
        x = ctx.obj["x"][1]
    except KeyError:
        x = np.arange(len(signal))
    if output is not None:
        out_path = Path(str(output))
        out = open(out_path, 'w+')
    else:
        out = click.get_text_stream('stdout')
    transformed_signal = list(trapezoid_filter(k, l, m, signal))
    for x_el, sig_el in zip(x, transformed_signal):
        out.write(f"{x_el},{sig_el}\n")
    out.close()


@click.command()
@click.pass_context
def plot(ctx: click.Context):
    x = ctx.obj["x"][1]
    y = ctx.obj["y"][1]
    plt.plot(list(x), list(y))
    plt.show()


file_io.add_command(read_csv)
