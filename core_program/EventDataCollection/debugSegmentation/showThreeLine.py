import matplotlib.pyplot as plt


# 数据提取函数
def extract_data(file_path):
    positive_events = []
    negative_events = []
    total_events = []

    with open(file_path, 'r') as f:
        for line in f:
            # 分割每一行，提取 Positive 和 Negative Events 的数量
            parts = line.split(', ')
            pos_str = parts[2].split(': ')[1]  # Positive Events: <number>
            neg_str = parts[3].split(': ')[1].strip()  # Negative Events: <number>

            pos = int(pos_str)
            neg = int(neg_str)
            total = pos + neg

            positive_events.append(pos)
            negative_events.append(neg)
            total_events.append(total)

    return positive_events, negative_events, total_events


# 绘图函数
def plot_events(positive_events, negative_events, total_events):
    # 创建三张独立的图
    frames = range(len(positive_events))  # X轴为帧序号

    # 图1：Positive Events
    plt.figure(figsize=(10, 6))
    plt.plot(frames, positive_events, label='Positive Events', color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Positive Events')
    plt.title('Positive Events Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('raw_positive_events.png')
    plt.close()

    # 图2：Negative Events
    plt.figure(figsize=(10, 6))
    plt.plot(frames, negative_events, label='Negative Events', color='blue')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Negative Events')
    plt.title('Negative Events Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('raw_negative_events.png')
    plt.close()

    # 图3：Total Events
    plt.figure(figsize=(10, 6))
    plt.plot(frames, total_events, label='Total Events', color='green')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Total Events')
    plt.title('Total Events Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('raw_total_events.png')
    plt.close()


# 主程序
if __name__ == "__main__":
    # 假设你的 TXT 文件路径为 'event_stats.txt'
    file_path = 'D:\\eventVision\\collecting\\3_3\\result\\raw\\event_stats.txt'  # 请修改为你的实际文件路径

    # 提取数据
    positive_events, negative_events, total_events = extract_data(file_path)

    # 绘制图形并保存
    plot_events(positive_events, negative_events, total_events)

    print("图形已生成并保存为 'positive_events.png', 'negative_events.png', 和 'total_events.png'")