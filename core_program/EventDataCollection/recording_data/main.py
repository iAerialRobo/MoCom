import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import argparse
import numpy as np

# 需要改四个地方
# Example usage command:
# python3 test_davis/writer_mono_dvs.py -c DAVIS346_00001074 -o ./test_davis/data/getAllDataFromCamera_1.aedat4
parser = argparse.ArgumentParser(description='Record data from a single iniVation camera to a file.')
parser.add_argument("-c",
                    "--camera_name",
                    dest='camera_name',
                    default="DAVIS346_00001074",
                    type=str,
                    help="Camera name (e.g. DVXplorer_DXA00093). The application will open any supported camera "
                    "if no camera name is provided.")
parser.add_argument("-o",
                    "--output_path",
                    dest='output_path',
                    default="./data/getAllDataFromCamera_3.aedat4",
                    type=str,
                    help="Path to an output aedat4 file for writing.")
args = parser.parse_args()
# this is the new board. 
pathPre = "D:\\eventVision\\collecting\\4_23"  # pre set event camera.

args.output_path = "D:\\eventVision\\collecting\\4_23\\3.aedat4"
args.camera_name = "DVXplorer_DXUS0002"
# Open any camera that is discovered in the system
camera = dv.io.CameraCapture(args.camera_name)

# Check whether frames are available
eventsAvailable = camera.isEventStreamAvailable()
framesAvailable = camera.isFrameStreamAvailable()
imuAvailable = camera.isImuStreamAvailable()
triggersAvailable = camera.isTriggerStreamAvailable()

i = 0

def slicing_callback(events: dv.EventStore):
    global i
    frame = visualizer.generateImage(events)
    length = len(events)
    event_values = np.array([event.polarity() for event in events])
 #   event_values = event_values.astype(int)
    #for event in events:
    #   print(event.time(), " ",type(event.time()))
    #   print(event.timestamp()," ", type(event.timestamp()))
    #   print(dir(event))
    #   c = 1
    timestamps = np.array([event.timestamp() for event in events])
  #  mean_value = np.mean(event_values)
   # variance_value = np.var(event_values)
   # std_deviation = np.std(event_values)
   # min_value = np.min(event_values)
   # max_value = np.max(event_values)
    if timestamps.size > 0:
        min_t, max_t = np.min(timestamps), np.max(timestamps)
        output_content = (f"{i},{min_t},{max_t},{length}\n")
        # 格式化输出内容
        # output_content = (f"{i},{min_t},{max_t},{mean_value},{variance_value},{std_deviation},{min_value},{max_value},{length}\n")

        # 保存统计信息到 txt 文件（追加写入）
        with open("D:\\eventVision\\collecting\\4_23\\3\\UAV_event_statistics.txt", "a") as f:
            f.write(output_content)

    else:
        output_content = (f"{i},{0},{0},{0}\n")
        # 格式化输出内容
        # output_content = (f"{i},{min_t},{max_t},{mean_value},{variance_value},{std_deviation},{min_value},{max_value},{length}\n")

        # 保存统计信息到 txt 文件（追加写入）
        with open("D:\\eventVision\\collecting\\4_23\\3\\UAV_event_statistics.txt", "a") as f:
            f.write(output_content)
        print("Array is empty!")


    cv.imwrite("D:\\eventVision\\collecting\\4_23\\3\\UAV_"+str(i)+".png",frame)
    i = i + 1
    #cv.imshow("Preview", frame)
    #cv.waitKey(33)

slicer = dv.EventStreamSlicer()
# cv.namedWindow("Preview", cv.WINDOW_NORMAL)
visualizer = dv.visualization.EventVisualizer(camera.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.white())
visualizer.setPositiveColor(dv.visualization.colors.red())
visualizer.setNegativeColor(dv.visualization.colors.green())
slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)

try:
    # Open a file to write, will allocate streams for all available data types
    writer = dv.io.MonoCameraWriter(args.output_path, camera)

    print("Start recording")
    while camera.isConnected():
        if eventsAvailable:
            # Get Events
            events = camera.getNextEventBatch()
            # Write Events
            if events is not None:
                writer.writeEvents(events, streamName='events')
                slicer.accept(events)

        if framesAvailable:
            # Get Frame
            frame = camera.getNextFrame()
            # Write Frame
            if frame is not None:
                writer.writeFrame(frame, streamName='frames')

        if imuAvailable:
            # Get IMU data
            imus = camera.getNextImuBatch()
            # Write IMU data
            if imus is not None:
                writer.writeImuPacket(imus, streamName='imu')

        if triggersAvailable:
            # Get trigger data
            triggers = camera.getNextTriggerBatch()
            # Write trigger data
            if triggers is not None:
                writer.writeTriggerPacket(triggers, streamName='triggers')

except KeyboardInterrupt:
    print("Ending recording")
    pass
