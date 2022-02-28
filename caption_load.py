"""
ĐÂY LÀ FOLDER XỬ LÝ TEXT, ĐÃ TÁCH RA KHÔNG CẦN PHẢI NHÚNG BÊN HÌNH ẢNH VÀO
"""

caption_path = r'C:\Users\Pham Thien An\PycharmProjects\data\token.txt'


# Hàm dùng để lấy caption
def load_captions(filename):
    with open(filename, "r") as fp:
        # đọc hết text xong file
        text = fp.read()
        return text


# hàm dùng để tạo dictionary
def captions_dict(text):
    dict = {}

    # Make a List of each line in the file
    lines = text.split('\n')
    for line in lines:

        # Split into the <image_data> and <caption>
        line_split = line.split('\t')
        if (len(line_split) != 2):
            # Added this check because dataset contains some blank lines
            continue
        else:
            image_data, caption = line_split

        # Split into <image_file> and <caption_idx>
        image_file, caption_idx = image_data.split('#')
        # Split the <image_file> into <image_name>.jpg
        image_name = image_file.split('.')[0]

        # If this is the first caption for this image, create a new list for that
        # image and add the caption to it. Otherwise append the caption to the
        # existing list
        if (int(caption_idx) == 0):
            dict[image_name] = [caption]
        else:
            dict[image_name].append(caption)

    return (dict)


doc = load_captions(caption_path)  # lấy caption
image_dict = captions_dict(doc)  # gán vào image_dict <(")
