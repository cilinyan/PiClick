import matplotlib.pyplot as plt

# 生成示例图形
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [2, 4, 3])
ax.set_title('Example Plot')

# 获取标题位置
pos = ax.title.get_position()

# 在标题位置添加新的文本
ax.text(pos[0], pos[1], 'New Text', ha='center', va='center', transform=ax.transAxes)

plt.show()