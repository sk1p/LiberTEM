class UDFException(Exception):
    """
    Raised when the UDF interface is somehow misused
    """
    pass


class AcquisitionCancelled(Exception):
    """
    Raised when the data stream received is cancelled before the expected size
    was reached.
    """
    pass
