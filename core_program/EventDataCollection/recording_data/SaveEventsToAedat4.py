import csv
import os.path
import cv2
import dv_processing as dv
import logging
import sys
import cv2 as cv
import numpy as np
from datetime import timedelta


def events_to_aedat4_file(
        events: dv.EventStore,
        resolution: tuple = (640, 480),
        file_name: str = 'cropped.aedat4'
) -> None:
    """
    Saves the given events to an aedat4 file.

    :param events: An event store
    :param resolution: A tuple specifying the resolution (width and height in pixels) of the given events.
    :param file_name: The file name of the generated aedat4 file.

    :return: None
    """
    config = dv.io.MonoCameraWriter.EventOnlyConfig(cameraName="DVXplorer_DXUS0002", resolution=resolution)
    writer = dv.io.MonoCameraWriter(file_name, config)
    writer.writeEvents(events)



# Generate 10 events with time range [10000; 20000]
store = dv.data.generate.uniformEventsWithinTimeRange(10000, timedelta(milliseconds=90000), (640, 480), 100000)

# Get all events beyond and including index 5
events_after_index = store.slice(5)
print(f"1. {events_after_index}")
events_to_aedat4_file(events_after_index, file_name='target.aedat4')
# Get 3 events starting with index 2
events_in_range = store.slice(2, 3)
print(f"2. {events_in_range}")

# Use sliceBack to retrieve event from the end; this call will retrieve last 3 events
last_events = store.sliceBack(3)
print(f"3. {last_events}")