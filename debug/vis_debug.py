import matplotlib.pyplot as plt

# 创建一个简单的图形
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

# 添加标题
ax.set_title('A Simple Plot')

# 获取标题的位置
title_pos = ax.title.get_position()

# 在标题位置添加文字
ax.text(title_pos[0], title_pos[1], 'This is the title of the plot.', ha='center', va='center', transform=ax.transAxes)

# 显示图形
plt.show()

"""
import matplotlib.pyplot as plt

# 创建一个简单的图形
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

# 添加标题
ax.set_title('A Simple Plot', fontsize=20, fontweight='bold')

# 获取标题对象
title_obj = ax.title

# 获取标题的位置和文本框
title_pos = title_obj.get_position()
title_bbox = title_obj.get_window_extent(renderer=fig.canvas.get_renderer())

# 在标题位置添加文本
ax.text(title_bbox.x0 + title_bbox.width/2., title_bbox.y1 + 0.05, 'This is another text.', ha='center', va='center', transform=ax.transAxes, fontsize=title_obj.get_fontsize(), fontweight=title_obj.get_fontweight(), fontfamily=title_obj.get_fontfamily(), color=title_obj.get_color())

# 显示图形
plt.show()

"""
