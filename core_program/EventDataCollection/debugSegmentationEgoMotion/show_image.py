import dv_processing as dv
import cv2 as cv
from datetime import timedelta


# filePath = 'D:\\workspace\\worksEventUtils\\4_27\\UAV_action_5.aedat4'
filePath = 'E:\\IEEE_tro_compareExperiment\\11_13\\recording_20251113_133240.aedat4'
# Open a file
reader = dv.io.MonoCameraRecording(filePath)

# Get and print the camera name that data from recorded from
print(f"Opened an AEDAT4 file which contains data from [{reader.getCameraName()}] camera")


def slicing_callback(events: dv.EventStore):
    frame = visualizer.generateImage(events)
    print(frame)
    cv.imshow("Preview", frame)
    cv.waitKey(33)

slicer = dv.EventStreamSlicer()
cv.namedWindow("Preview", cv.WINDOW_NORMAL)
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.white())
visualizer.setPositiveColor(dv.visualization.colors.red())
visualizer.setNegativeColor(dv.visualization.colors.green())
slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)

# Collect all events into a list
all_events_list = []
while reader.isRunning():
    events = reader.getNextEventBatch()
    if events is not None:
        for event in events:  # Iterate over events in the batch
            all_events_list.append(event)

# Sort the list by timestamp
all_events_list.sort(key=lambda e: e.timestamp())

# Reconstruct a sorted EventStore
#sorted_store = dv.EventStore()
#or e in all_events_list:
    #sorted_store.add(dv.Event(e.timestamp(), e.x(), e.y(), e.polarity()))

sorted_store = dv.EventStore()
for e in all_events_list:
    sorted_store.push_back(e.timestamp(), e.x(), e.y(), e.polarity())

# Now accept the whole sorted store into the slicer
slicer.accept(sorted_store)

# Wait for key press to exit (optional, to keep window open)
cv.waitKey(0)
cv.destroyAllWindows()