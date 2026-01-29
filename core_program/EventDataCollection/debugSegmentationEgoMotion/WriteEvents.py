import re

# === 1. 设置输入与输出路径 ===
log_path = r"E:\IEEE_tro_compareExperiment\11_12_afternoon\out_putFile\detection_log_20251112_171503.txt"
output_path = r"./filtered_frame_data_650_2400.txt"

# === 2. 定义正则表达式匹配 ===
pattern = re.compile(
    r"Frame\s+(\d+).*?"
    r"Total Events:\s+(\d+).*?"
    r"Positive Events:\s+(\d+).*?"
    r"Negative Events:\s+(\d+).*?"
    r"Pos Ratio:\s+([0-9.]+)",
    re.S
)

# === 3. 读取文件并匹配 ===
frames, pos_events, neg_events = [], [], []

with open(log_path, 'r', encoding='utf-8') as f:
    text = f.read()
    matches = pattern.findall(text)

for m in matches:
    frame = int(m[0])
    pos = int(m[2])
    neg = int(m[3])

    # === 只取 Frame 在 650 ~ 2400 之间的数据 ===
    if 650 < frame <= 2400:
        frames.append(frame)
        pos_events.append(pos)
        neg_events.append(neg)

# === 4. 写入新文件 ===
with open(output_path, 'w', encoding='utf-8') as f_out:
    for i, frame in enumerate(frames):
        line = (
            f"Frame {i:06d}: Start Time: 0, End Time: 0, "
            f"Positive Events: {pos_events[i]}, Negative Events: {neg_events[i]}\n"
        )
        f_out.write(line)

print(f"已保存到文件: {output_path}")
print(f"共写入 {len(frames)} 行 (Frame 651 ~ 2400)")
