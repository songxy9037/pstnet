import numpy as np
from PIL import Image

def visualize_attention_map(attention_map):
    """
    The attention map is a matrix ranging from 0 to 1, where the greater the value,
    the greater attention is suggests.
    :param attention_map: np.numpy matrix hanging from 0 to 1
    :return np.array matrix with rang [0, 255]
    """
    # save_path = ''
    # attention_map /= 255
    attention_map = attention_map.permute(0,2,3,1)
    attention_map = attention_map.cpu().numpy()
    attention_map = attention_map.squeeze()
    print(attention_map.shape)
    # attention_map = np.linalg.norm(attention_map)
    attention_map_color = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1], 3],
        dtype=np.uint8
    )

    red_color_map = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1]],
        dtype=np.uint8
    ) + 255

    red_color_map = red_color_map * attention_map
    red_color_map = np.array(red_color_map, dtype=np.uint8)

    attention_map_color[:, :, 2] = red_color_map

    # att_save = Image.fromarray(attention_map)
    # att_save.save(save_path)
    return attention_map_color
