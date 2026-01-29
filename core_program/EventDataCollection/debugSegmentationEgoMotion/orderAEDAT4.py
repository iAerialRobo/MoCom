import dv_processing as dv
import os

# === 根目录路径 ===
# root_dir = r"E:\IEEE_tro_compareExperiment\11_13"

root_dir = r"E:\IEEE_tro_compareExperiment\11_14"
# === 遍历 1, 2, 3 子文件夹 ===
for folder_name in ["1", "2", "3"]:
    folder_path = os.path.join(root_dir, folder_name)

    # 找出子文件夹下的 .aedat4 文件（假设只有一个）
    aedat_files = [f for f in os.listdir(folder_path) if f.endswith(".aedat4")]
    if not aedat_files:
        print(f"{folder_path} 下未找到 .aedat4 文件，跳过。")
        continue

    input_path = os.path.join(folder_path, aedat_files[0])
    output_path = os.path.join(folder_path, "order.aedat4")

    print(f"\n处理文件: {input_path}")

    # === 读取 aedat4 文件 ===
    reader = dv.io.MonoCameraRecording(input_path)
    print(f"摄像头: {reader.getCameraName()}")

    # === 读取所有事件 ===
    all_events = []
    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is not None:
            for e in events:
                all_events.append(e)

    print(f"读取事件总数: {len(all_events)}")

    # === 按时间戳排序 ===
    all_events.sort(key=lambda e: e.timestamp())

    # === 构造新的 EventStore ===
    sorted_store = dv.EventStore()
    for e in all_events:
        sorted_store.push_back(e.timestamp(), e.x(), e.y(), e.polarity())

    resolution = (640, 480)

    # Event only configuration
    config = dv.io.MonoCameraWriter.EventOnlyConfig("DVXplorerM_DXUS0002", resolution)

    # Create the writer instance, it will only have a single event output stream.
    # writer = dv.io.MonoCameraWriter("mono_writer_sample.aedat4", config)

    # === 写入新的 aedat4 文件 ===
    writer = dv.io.MonoCameraWriter(output_path, config)
    writer.writeEvents(sorted_store)
    # writer.close()

    print(f"已保存排序后的文件: {output_path}")

print("\n全部处理完成！")
