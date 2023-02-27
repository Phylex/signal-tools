from pathlib import Path
import pandas as pd
import numpy as np
import click


@click.group()
@click.option("-i", "--input", type=click.Path(dir_okay=False, exists=True))
@click.option("-c", "--signal-column",
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
def cli(ctx, input: click.Path, signal_column: int, filetype: str,
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


@cli.command()
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


@cli.command()
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


@cli.command()
@click.argument("lsb-magnitude", type=float)
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Specify a file to write the output of the command to. If not "
              "specified, 'stdout' will be used")
@click.pass_context
def digitize(ctx: click.Context, lsb_magnitude: float, output: click.Path) -> None:
    y = ctx.obj["y"]
    try:
        x = ctx.obj["x"]
    except KeyError:
        x = ("Measurement Samples", np.arange(len(y[1])))
    measurements = y[1]
    if isinstance(measurements, np.ndarray):
        measurements: np.ndarray
        digitized_meas = np.apply_along_axis(lambda x: x // lsb_magnitude, 0, measurements)
    else:
        measurements
    ...
