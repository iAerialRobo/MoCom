import dv_processing as dv
from datetime import timedelta



def events_to_aedat4_file(
        events: dv.EventStore,
        resolution: tuple = (100, 100),
        file_name: str = 'cropped.aedat4'
) -> None:
    """
    Saves the given events to an aedat4 file.

    :param events: An event store
    :param resolution: A tuple specifying the resolution (width and height in pixels) of the given events.
    :param file_name: The file name of the generated aedat4 file.

    :return: None
    """
    config = dv.io.MonoCameraWriter.EventOnlyConfig(cameraName="DAVIS346_00000305", resolution=resolution)
    writer = dv.io.MonoCameraWriter(file_name, config)
    writer.writeEvents(events)


# 生成事件数据
store = dv.data.generate.uniformEventsWithinTimeRange(10000, timedelta(milliseconds=10), (100, 100), 10)
# writer = dv.io.EventOutput("output.aedat4")
# 创建 AEDAT 4 文件并写入事件
# with dv.io.Aedat4Writer("event_store.aedat4") as writer:
#     writer.writeEvents(store)  # 直接写入 eventStore 对象

# print("事件数据已成功保存到 event_store.aedat4")

print(dir(dv.io))

# 创建 .aedat4 文件的写入器
# writer = dv.io.OutputEventFile("event_store.aedat4")

# 写入事件
# writer.write(store)

# 关闭文件
# writer.close()

print("事件数据已成功保存到 event_store.aedat4")