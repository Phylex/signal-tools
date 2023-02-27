from itertools import tee
from pathlib import Path
import pandas as pd
import numpy as np
import click
import matplotlib.pyplot as plt 

from signal_tools.trapezfilter import trapezoid_filter


@click.group()
@click.argument("input", type=click.Path(dir_okay=False, exists=True))
@click.option("-s", "--signal-column",
              type=click.IntRange(min=0, max_open=True),
              help="column of the data that should be used as the signal")
@click.option("-f", "--filetype", type=click.Choice(["csv", "h5", "stream"]),
              default="csv",
              help="Type of file to be read in")
@click.option("-d", "--delimiter", type=str, default=",",
              help="item delimiter, only used with csv format")
@click.option("-x", "--x-column", type=click.IntRange(min=0, max_open=True),
              default=None,
              help="Column in the Data used for the x-axis")
@click.pass_context
def file_cli(ctx, input: click.Path,
             signal_column: int, filetype: str,
             delimiter: str,
             x_column: int) -> None:
    ctx.obj = {}
    input_path = Path(str(input))
    match filetype:
        case "csv":
            if input_path.suffix != "." + filetype:
                click.echo(
                    "Warning, file ending does not match the specified ending")
            input_data = open(input_path, 'r')
            i_data: pd.DataFrame = pd.read_csv(input_data, delimiter=delimiter)
            signal_data: np.ndarray = i_data.iloc[:, signal_column].to_numpy()
            signal_name: str = i_data.iloc[:, signal_column].name
            if x_column is not None:
                x_col = i_data.iloc[:, x_column].to_numpy()
                x_name = i_data.iloc[:, x_column].name
                ctx.obj["x"] = (x_name, x_col)
        case _:
            click.echo("Filetype not implemented")
            return
    ctx.obj["y"] = (signal_name, signal_data)


@click.group()
@click.option("-s", "--signal-column", type=click.IntRange(0, max_open=True),
              default=1)
@click.option("-x", "--x-column", type=click.IntRange(0, max_open=True),
              default=0)
@click.option("-d", "--delimiter", type=click.Choice([",", ";", ":"]),
              default=",")
@click.pass_context
def stream_cli(ctx: click.Context, signal_column: int, x_column: int,
               delimiter: str):
    """
    Read in data from stdin and pass on to filter
    """
    in_stream = click.get_text_stream('stdin')
    in_0, in_1 = tee(in_stream, 2)
    x_data = map(lambda line: float(line.split(delimiter)[x_column].strip()), in_0)
    signal = map(lambda line: float(line.split(delimiter)[signal_column].strip()), in_1)
    ctx.obj = {"x": ("Samples", x_data), "y": ("Signal", signal)}


@click.command()
@click.pass_context
def to_stream(ctx: click.Context) -> None:
    """
    convert the file read in into a stream
    """
    output = click.get_text_stream("stdout")
    _, sig_data = ctx.obj["y"]
    try:
        _, x_data = ctx.obj["x"]
        for x_d, sig_d in zip(x_data, sig_data):
            output.write(f"{x_d}, {sig_d}\n")
    except KeyError:
        for sig_d in sig_data:
            output.write(f"{sig_d}\n")
    return


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

file_cli.add_command(to_stream)
file_cli.add_command(print_input)
file_cli.add_command(digitize)
file_cli.add_command(apply_trapezoidal_filter)
