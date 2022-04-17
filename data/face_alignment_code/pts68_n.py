import numpy as np


def read_file(file_name):
    res = []
    with open(file_name) as fp:
        for line in fp:
            res.append(line.strip(' ').strip('\n').strip('\r'))
    return res


def average_point(num1, num2, rescale_arr):
    x1 = rescale_arr[num1 * 2 - 2]
    y1 = rescale_arr[num1 * 2 - 1]
    x2 = rescale_arr[num2 * 2 - 2]
    y2 = rescale_arr[num2 * 2 - 1]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    return x, y


def change68to24(path_68, path_n):
    pts_68_list = read_file(path_68)
    item_length = len(pts_68_list)
    final_points = np.zeros([item_length, 48], np.int32)
    point_num = 0

    for index, line in enumerate(pts_68_list):
        land_mark = line.strip(' ').split(' ')
        rescale_arr = np.asarray(land_mark, np.float32)
        # collect reslt
        single = []
        # au1: 22, 23
        point = [19, 22, 23, 26, 39, 37, 44, 46, 28, 30, 49, 51, 53, 55, 59, 57]
        double_point = [
            [20, 38],
            [25, 45],
            [41, 42],
            [47, 48],
            [18, 59],
            [27, 57]
        ]
        for ele in point:
            single.append(rescale_arr[ele * 2 - 2])
            single.append(rescale_arr[ele * 2 - 1])  # 6 point

        for tmp_idx in range(0, 4):
            # denote
            # single[tmp_idx*2+1] = single[tmp_idx*2+1] - 20
            single[tmp_idx * 2 + 1] = single[tmp_idx * 2 + 1] - 10

            # refine 19, 26 idx = 0, 3
            single[0 * 2 + 1] = single[0 * 2 + 1] + 2
            single[3 * 2 + 1] = single[3 * 2 + 1] + 2

        for ele in double_point:
            x, y = average_point(ele[0], ele[1], rescale_arr)
            single.append(x)
            single.append(y)  # 7 point

        single.append(rescale_arr[49 * 2 - 2] - 16)
        single.append(rescale_arr[49 * 2 - 1] - 16)  # 15 point
        single.append(rescale_arr[55 * 2 - 2] + 16)
        single.append(rescale_arr[55 * 2 - 1] - 16)  # 16 point
        point_num = len(single) / 2

        result_arr = np.asarray(single, np.float32)
        rescale_single = result_arr + 0.5
        rescale_single = rescale_single.astype(int)
        final_points[index, :] = rescale_single
        # final_points[idx,:] = result_arr

    with open(path_n, 'w') as fp:
        for idx in range(0, item_length):
            point = final_points[idx]
            for inner_idx in range(0, int(point_num) * 2):
                fp.write(str(point[inner_idx]))
                if inner_idx < point_num * 2 - 1:
                    fp.write(' ')
            fp.write('\n')


def change68to16(path_68, path_n):
    pts_68_list = read_file(path_68)
    item_length = len(pts_68_list)
    final_points = np.zeros([item_length, 32], np.int32)
    point_num = 0

    for index, line in enumerate(pts_68_list):
        land_mark = line.strip(' ').split(' ')
        rescale_arr = np.asarray(land_mark, np.float32)
        # collect reslt
        single = []
        # au1: 22, 23
        point = [19, 22, 23, 26, 28, 30, 49, 55, 52, 58]
        double_point = [
            [41, 42],
            [47, 48],
            [18, 59],
            [27, 57]
        ]
        for ele in point:
            single.append(rescale_arr[ele * 2 - 2])
            single.append(rescale_arr[ele * 2 - 1])  # 6 point

        for tmp_idx in range(0, 4):
            # denote
            # single[tmp_idx*2+1] = single[tmp_idx*2+1] - 20
            single[tmp_idx * 2 + 1] = single[tmp_idx * 2 + 1] - 10

            # refine 19, 26 idx = 0, 3
            single[0 * 2 + 1] = single[0 * 2 + 1] + 2
            single[3 * 2 + 1] = single[3 * 2 + 1] + 2

        for ele in double_point:
            x, y = average_point(ele[0], ele[1], rescale_arr)
            single.append(x)
            single.append(y)  # 7 point

        single.append(rescale_arr[49 * 2 - 2] - 16)
        single.append(rescale_arr[49 * 2 - 1] - 16)  # 15 point
        single.append(rescale_arr[55 * 2 - 2] + 16)
        single.append(rescale_arr[55 * 2 - 1] - 16)  # 16 point
        point_num = len(single) / 2

        result_arr = np.asarray(single, np.float32)
        result_arr = result_arr * 28 / 224
        rescale_single = result_arr + 0.5
        rescale_single = rescale_single.astype(int)
        final_points[index, :] = rescale_single
        # final_points[idx,:] = result_arr

    with open(path_n, 'w') as fp:
        for idx in range(0, item_length):
            point = final_points[idx]
            for inner_idx in range(0, int(point_num) * 2):
                fp.write(str(point[inner_idx]))
                if inner_idx < point_num * 2 - 1:
                    fp.write(' ')
            fp.write('\n')


if __name__ == '__main__':
    path_68 = './68pts.list'
    path_n = './16pts.list'
    # change68to24(path_68, path_n)
    change68to16(path_68, path_n)
