def normalize_bbox(x, y, w, h, img_width, img_height):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    return x_center, y_center, norm_w, norm_h




