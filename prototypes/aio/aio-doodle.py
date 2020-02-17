"""
What: a general async I/O layer for the kind of I/O
that LiberTEM does (mostly sequential, large blocks,
most I/O of the same size, ...)

- Fixed number of I/O buffers
- Compute engine needs to explicitly tell I/O engine
  when it is done with a buffer (don't wait for GC...)
- Back pressure: I/O engine only creates as many I/O
  requests as there are free I/O buffers.
- Can more easily detect if I/O or compute is the bottleneck,
  depending on number of in-flight requests

- For live acquisition, this needs to be adjusted: we don't
  want to drop frames in most situations. Allow the
  "buffer pool" to grow, but of course bounded by
  the available memory.
  Also: if acquisition is too fast for compute,
  switch to semi-live mode and do the processing
  from mass storage.
- Buffers should be bounded by some maximal value;
  and because we don't want to waste memory most
  requests should be of this maximum size
  (i.e. one that maps to the "tilesize" of the
  higher layers of LiberTEM)

Where does this I/O code interface with the rest of LiberTEM?
- Mostly in the DataSet impls, which should become a thin
  layer over this one, adding decoding and some
  dataset specifics.
- Decoding and other dataset specific compute should
  happen in the same process and "execution flow" as
  the UDF compute.

- We need to try and fit this into a contextmanager,
  to make it easy to use it correctly. Should be passed
  down to the "end user", i.e. UDFRunner

Use cases for AIO:
- 1) "Normal" offline processing from "disk" (may beed direct I/O and liburing)
- 2) Reading from TCP/HTTP (I/O thread should be sufficient, slow data rates expected)
- 3) Online acquisition and live processing
     (high data rates: direct I/O is important again!)

Differences between the use cases:
- 2 and 3 are mostly guaranteed sequential, where 1 can be out of order. 
- 3 can also be an "unlimited" stream that is not written do disk, but just
  used i.e. as feedback signal either for the user or for an algorithm.
- 2 and 3 are very much driven by whatever comes from the detector, where
  1 can do things like ROI in navigation dimensions etc.
"""


class Buffer:
    """
    Could be a long-living objects representing a buffer slot.
    """
    def __init__(self, buffer_manager):
        self.buffer_manager = buffer_manager

    def __enter__(self):
        # FIXME: maybe gate data availability enter/exit, so the users don't
        # use a Buffer without marking it available again
        pass

    def __exit__(self, *exc):
        self.buffer_manager.mark_available(self)


class Partition:
    def make_read_ranges(self, tile_spec):
        """
        generator that yields ranges as understood by the AIOEngine
        """
        raise NotImplementedError()

    def get_aio_engine(self):
        raise NotImplementedError()

    def get_tiles(self, tile_spec):
        ranges = self.make_read_ranges(tile_spec)
        buffer_manager = self.get_buffer_manager()
        aio_engine = self.get_aio_engine(buffer_manager)
        # aoi engine can take as many "read ranges" from the iterator
        # as it needs, when it needs them
        for range_, buf in aio_engine.read_ranges(ranges):
            # range_ identifies which part of the data we have available,
            # buf is the raw data that was read
            with buf:
                # in the easiest case, `tile` may directly reference `buf`
                # (i.e. a view into buf)
                tile = self.decode(range_, buf)
                yield tile  # compute happens on the "receiver" end of the generator
            # buf now free again for the next read request


class BufferManager:
    """
    Three buffer states:
    - available: this buffer can be used for reading
    - reading: there is currently a read request in flight for this buffer
    - full: there is data in this buffer which has not been used yet

    Buffers start in the `available` state, transition to `reading`,
    where they stay until they contain the data read from the source.
    When the data has been read, the buffer transitions to the `full`
    state and stays there until the data has been processed by the
    associated
    """
    def __init__(self, number, size):
        pass

    def get_available(self):
        raise NotImplementedError()

    def get_reading(self):
        raise NotImplementedError()

    def mark_available(self, buf):
        """
        full → available
        """
        raise NotImplementedError()


    def mark_start_reading(self, buf):
        """
        available → reading
        """
        raise NotImplementedError()

    def mark_reading_done(self, buf):
        """
        reading → full
        """
        raise NotImplementedError()


class AIOEngine:
    def __init__(self, buffer_manager):
        self.buffer_manager = buffer_manager

    def get_completions(self):
        """
        """
        pass

    def read_ranges(self, ranges):
        """
        """
        # all I/O reqeusts have been _submitted_:
        submissions_done = False

        while not submissions_done:
            for buf in self.buffer_manager.get_available():
                try:
                    range_ = next(ranges)
                    self.add_read_request(range_, buf)
                except StopIteration:
                    submissions_done = True
            for range_, buf in self.get_completions():
                yield range_, buf

        while len(self.buffer_manager.get_reading()) > 0:
            for range_, buf in self.get_completions():
                yield range_, buf


def example_compute_load(partition):
    """
    compute load stays the same!

    so UDFRunner also doesn't need to change
    """
    res = np.zeros(partition.shape.sig)
    for tile in partition.get_tiles():
        res += tile.sum(axis=0)



def example_live_processing():
    """
    Process live data as it comes from the detector, for example
    to generate a monitoring signal or a signal that will
    decide when acquisition should start.

    "continuous" - data is not written by disk.
    """


def example_online_acquisition():
    """
    Process a bounded stream of data, for example for a given
    scan, or time period, while writing the data to disk.

    As opposed to live processing, here we can also create 2D maps,
    which is not possible with an unbounded signal.

    Possible to change parameters while acquisition is running,
    which will re-process the already acquired data from disk.
    """

    camera.set_parameters({
        "...": ...,  # FIXME: whatever parameters are needed
    })
    camera.prime()
    microscope.set_parameters({
        "...": ...,  # FIXME: whatever parameters are needed
    })

    udf = ApplyMaskUDF(masks=[lambda: np.ones((256, 256))])
    acquisition = ctx.prepare_acquisition(name="some_acquisition", udf=udf)

    result = acquisition.get_result_arr()

    # FIXME: wrap matplotlib?
    the_plot = plt.imshow(acquisition.result)

    microscope.start_acquisition()

    for current_result in acquisition.run():
        the_plot.set_data(current_result)
        the_plot.set_clim(vmin=..., vmax=...)
        plt.draw()

