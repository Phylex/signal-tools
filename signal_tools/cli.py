from itertools import tee
from pathlib import Path
import sys
import csv
import re
import numpy as np
import click
import matplotlib.pyplot as plt
from signal_tools.filter import trapezoid_filter


@click.group("signal-io")
@click.argument("file-path", type=click.Path(dir_okay=False))
@click.argument("direction", type=click.Choice(["in", "out", "append"], case_sensitive=False))
@click.argument("encoding", type=click.Choice(["bin", "utf-8"], case_sensitive=False))
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
@click.option("-s", "--signal-column", type=int, multiple=True, required=True,
              help="index of the column that should be read in")
@click.pass_context
def read_csv(ctx, delimiter, signal_column):
    """
    read in a file of CSV format.
    """
    colname_regex = re.compile(
        '^(?P<colname>\w+([\t| ]+\w+)*)[\t| ]?(:(?P<type>[int|float|str][arr]?):)?\[(?P<prefix>[k|m|u|n|p|f|M|G|T|E|ki|Mi|Gi|Ti])?(?P<unit>[m|V|A|g|K|cd|s|N])\]$')
    csvr = csv.reader(
        ctx.obj['in'], delimiter=delimiter, skipinitialspace=True)
    out = ctx.obj['out']
    col_names = next(csvr)
    column_descriptions = []
    for i, name in enumerate(col_names):
        cmatch = colname_regex.match(name)
        if cmatch is None:
            click.echo(f"'{name}'has an invalid format")
            sys.exit(3)
        column_descriptions.append({'name': cmatch.group('colname'),
                                    'prefix': cmatch.group('prefix'),
                                    'unit': cmatch.group('unit'),
                                    'type': cmatch.group('type')
                                    })
    if len(col_names) < max(signal_column):
        click.echo("Column requested that does not exist")
        sys.exit(4)
    requested_cols_desc = [column_descriptions[i] for i in signal_column]

    # write the requested_cols to the data stream
    for col in requested_cols_desc:
        out.write(f"{col['name']}:")
        if col['type'] is not None:
            out.write(f"{col['type']}:")
        if col['prefix'] is not None:
            out.write(f"{col['prefix']}")
        if col['unit'] is not None:
            out.write(f"{col['unit']}")
        out.write(";")
    out.write('\n')

    # now write the stream info to the
    colcount = len(signal_column) - 1
    error = False
    for line in csvr:
        req_data = map(lambda elem: elem[1], filter(
            lambda elem: elem[0] in signal_column, enumerate(line)))
        for i, (elem, descr) in enumerate(zip(req_data, column_descriptions)):
            match descr['type']:
                case None:
                    out.write(elem + ";")
                case 'str':
                    out.write(elem + ";")
                case 'int':
                    try:
                        pelem = int(elem)
                        out.write(repr(pelem))
                    except ValueError:
                        out.write("0")
                        error = True
                        click.echo(
                            f"Error parsing integer on line {csvr.line_num}, column {signal_column[i]}")
                case 'float':
                    try:
                        pelem = float(elem)
                        out.write(repr(pelem))
                    except ValueError:
                        out.write("0.0")
                        error = True
                        click.echo(
                            f"Error parsing float on line {csvr.line_num}, column {signal_column[i]}")
                case _:
                    click.echo("Parser Failiure. Stopping")
                    sys.exit(4)
            if i < colcount:
                out.write(";")
        if error:
            sys.exit(5)
        out.write("\n")
    return


@click.group()
@click.pass_context
def stream_cli(ctx: click.Context):
    """
    Handle all the parsing and pass the data to the subcommands
    """
    in_stream = click.get_text_stream('stdin')
    metadata_line = in_stream.readline()
    stream_descriptors = metadata_line.split(";")
    descr_regex = re.compile(
        r'(?P<name>\w+([\t| ]+\w+)*)(:(?P<type>[int|float|str](arr)?))?:((?P<prefix>[k|m|u|n|p|f|M|G|T|E|ki|Mi|Gi|Ti])?(?P<unit>[m|V|A|g|K|cd|s|N]))?')
    meta_data = []
    for sdescr in stream_descriptors:
        m = descr_regex.match(sdescr)
        if m is None:
            click.echo("Malformed metadata line at beginning of stream")
            sys.exit(4)
        meta_data.append({'name': m.group('name'),
                          'prefix': m.group('prefix'),
                          'unit': m.group('unit'),
                          'type': m.group('type')}
                         )
    line_regex = re.compile(
        r'\[?([+|-]?[\d+|\d+\.\d+])[\],]?')

    def parsed_stream():
        for line in in_stream:
            columns = [re.findall(line_regex, elem)
                       for elem in line.split(';')]
            parsed_cols = []
            for elem, mdata in zip(columns, meta_data):
                match mdata['type']:
                    case 'int':
                        parsed_cols.append(list(map(int, elem))[0])
                    case 'float':
                        parsed_cols.append(list(map(float, elem))[0])
                    case 'intarr':
                        parsed_cols.append(list(map(int, elem)))
                    case 'floatarr':
                        parsed_cols.append(list(map(float, elem)))
            yield parsed_cols

    ctx.obj = {'data': parsed_stream, 'metadata': meta_data}


@click.command()
@click.pass_context
def print_input(ctx: click.Context) -> None:
    """
    Print the input read from the file. Make sure that the data has been
    properly read in by the toolbox
    """
    try:
        x = ctx.obj["x"]
        click.echo("Reference:")
        click.echo(x[0])
        click.echo(x[1])
    except KeyError:
        pass
    y = ctx.obj["y"]
    click.echo("\nSignal:")
    click.echo(y[0])
    click.echo(y[1])
    return


@click.command()
@click.argument("lsb-magnitude", type=float)
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Specify a file to write the output of the command to. If not "
              "specified, 'stdout' will be used")
@click.pass_context
def digitize(ctx: click.Context, lsb_magnitude: float, output: click.Path) -> None:
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


stream_cli.add_command(apply_trapezoidal_filter)
stream_cli.add_command(digitize)
stream_cli.add_command(plot)

file_io.add_command(print_input)
file_io.add_command(read_csv)
