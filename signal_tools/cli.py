from pathlib import Path
import sys
import csv
from click.types import IntRange
import yaml
import click
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import partial
from .stream_utils import arrays_to_data_line, collect_stream_into_string
from .parsers import SignalStreams
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
    ENCODING specifies if the file should be read as binary or utf-8 encoded
    file
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
        raise ValueError(f"Invalid column index in column selection. "
                         f"{len(col_names)} available")

    # generate the description dict for the requested data
    column_descriptions = []
    for i, name in enumerate(col_names):
        if i in signal_column:
            if i in columns_with_types:
                ct = column_type[i][1]
            else:
                ct = 'float'
            column_descriptions.append(
                {'name': str(name),
                 'type': deepcopy(ct),
                 'shape': [1]
                 })

    # write the meta data for the requested columns to the data stream
    out.write("Metadata:\n")
    out.write("streams:\n")
    out.write(yaml.dump(column_descriptions))
    out.write("\nData:\n")

    # now write the stream info to the
    for line in csvr:
        # filter out the columns that we wanted
        req_data = list(
                map(lambda elem: np.array([elem[1]],
                                          dtype=column_descriptions[elem[0]]['type']),
                    filter(lambda elem: elem[0] in signal_column,
                           enumerate(line))))
        out.write(arrays_to_data_line(req_data)+'\n')


@click.group("signal-transform")
@click.option("-v", "--verbose", count=True)
@click.option("-s", "--stream", type=click.IntRange(min=0, max_open=True), required=True,
              help="Select the stream that the command should be applied to")
@click.pass_context
def apply_transformation(ctx: click.Context, verbose: int, stream: int):
    """
    Apply a Transformation onto one of the data streams.

    This command prepares the data and lets the subcommands execute
    """
    stream_in = click.get_text_stream('stdin')
    data_stream = SignalStreams(stream_in)
    data_streams = data_stream.split_into_individual_streams()
    if verbose > 0:
        for i, (metadata, _) in enumerate(data_streams):
            click.echo(f"Stream {i}: {metadata['name']}")
    if stream > len(data_streams):
        click.echo(f"No stream with index {stream}. "
                   f"{len(data_streams)} streams available")
        sys.exit()
    ctx.obj = {}
    ctx.obj['streams'] = data_streams
    ctx.obj['selected_stream_idx'] = stream


@click.command()
@click.argument("lsb-magnitude", type=float)
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
    stream_idx = ctx.obj['selected_stream_idx']
    data_iterators = [st[1] for st in ctx.obj['streams']]
    metadata_copy = [deepcopy(ds[0]) for ds in ctx.obj['streams']]
    metadata_copy[stream_idx]['type'] = int

    def digitize_array(array: np.ndarray, lsb_mag: float) -> np.ndarray:
        array = array / lsb_mag
        return array.astype(int)

    transformed_stream = map(
            partial(digitize_array, lsb_mag=lsb_magnitude),
            data_iterators[stream_idx])
    data_iterators[stream_idx] = transformed_stream

    output_it = collect_stream_into_string(list(zip(metadata_copy, data_iterators)))
    for string in output_it:
        out.write(string)
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
@click.argument('y', type=click.IntRange(0, max_open=True))
@click.option('-m', '--mode',
              type=click.Choice(['xy', 'waterfall', 'matrix', 'scatter']),
              help="set the type of plot that will be produced")
@click.option('-x', '--xaxis', type=IntRange(0, max_open=True),
              help="Specify the X axis for which to plot which to plot")
@click.pass_context
def plot(ctx: click.Context, y: int, mode: str, xaxis: int) -> None:
    ...


file_io.add_command(read_csv)
apply_transformation.add_command(digitize)
