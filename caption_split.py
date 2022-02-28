"""
Chia ra để dùng cho train, validate và test
"""

training_image_name_file = r"C:\Users\Pham Thien An\PycharmProjects\data\trainImages.txt"


# ném file vào lấy tên ảnh ra
def subset_image_name(filename):
    data = []

    with open(filename, "r") as fp:
        #đọc file
        text = fp.read()

        # Split thành từng dòng
        lines = text.split('\n')
        for line in lines:
            # bỏ line rỗng
            if len(line) < 1:
                continue

            # Mỗi dòng trong image file
            # Split img_name.jpg thành img_name và jpg
            image_name = line.split('.')[0]

            # Add mỗi img_name vào
            data.append(image_name)

        return set(data)


training_image_names = subset_image_name(training_image_name_file)
