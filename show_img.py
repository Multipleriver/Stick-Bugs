import cv2
import os

# 获取当前工作目录
current_dir = os.getcwd()
# 构建图片的完整路径
image_path = os.path.join(current_dir, 'bus.jpg')

# 读取图片
img = cv2.imread(image_path)

# 检查图片是否成功读取
if img is None:
    print(f"无法读取图片，请确认 {image_path} 存在")
else:
    # 显示图片
    cv2.imshow('Bus Image', img)
    # 等待按键关闭窗口
    cv2.waitKey(0)
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
