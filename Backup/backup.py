def __calculate_max_gradient(gray_img, start_pos, distance=0, vertical=True, slope=0):
    """Get the max gradient in the slope direction and within the distance

    Arguments:
        gray_img {[type]} -- [description]
        start_pos {[type]} -- [description]

    Keyword Arguments:
        distance {int} -- [description] (default: {0})
        vertical {bool} -- [description] (default: {True})
        slope {int} -- [description] (default: {0})

    Returns:
        point_max_gradient {tuple} -- [description]
    """
    # x is from left to right, related to width
    # y is from up to down, related to height
    x, y = start_pos
    max_grad = -1
    grad_x, grad_y = 0, 0

    if vertical:
        # Let column(x) to be fixed and change the row(y)
        for dy in range(int(-distance / 2), int(distance / 2)):
            if __IsValid(x, y + dy + 1, gray_img) and __IsValid(x, y + dy, gray_img) and abs(gray_img[y + dy + 1, x] - gray_img[y + dy, x]) > max_grad:
                max_grad = abs(gray_img[y + dy + 1, x] - gray_img[y + dy, x])
                grad_x, grad_y = x, y + dy
    else:
        last_x, last_y = x, y
        c_x, c_y = x, y
        # up
        while __IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
            if abs(gray_img[c_y, c_x] - gray_img[last_y, last_x]) > max_grad:
                max_grad = abs(gray_img[c_y, c_x] - gray_img[last_y, last_x])
                grad_x, grad_y = c_x, c_y
            last_x, last_y = c_x, c_y
            c_x = c_x + 1
            c_y = int(c_y + 1 * slope)

        last_x, last_y = x, y
        c_x, c_y = x, y
        # down
        while __IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
            if abs(gray_img[c_y, c_x] - gray_img[last_y, last_x]) > max_grad:
                max_grad = abs(gray_img[c_y, c_x] - gray_img[last_y, last_x])
                grad_x, grad_y = c_x, c_y
            last_x, last_y = c_x, c_y
            c_x = c_x - 1
            c_y = int(c_y - 1 * slope)

    # Check the ans
    if max_grad == -1:
        return start_pos
    else:
        return (grad_x, grad_y)

