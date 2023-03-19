
def readchunks(stream: StringIO, delimiter: str, fio_size: int = 256):
    buf = ""
    while True:
        buf += stream.read(fio_size)
        parts = buf.split(delimiter)


