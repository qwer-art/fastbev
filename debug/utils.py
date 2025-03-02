from enum import Enum
import numpy as np
import quaternion
import cv2
from pyquaternion import Quaternion

np.set_printoptions(suppress=True,precision=4)
class Color(Enum):
    kRed = (0, 0, 255)
    kGreen = (0, 255, 0)
    kBlue = (255, 0, 0)
    kYellow = (0, 255, 255)
    kMagenta = (255, 0, 255)
    kCyan = (255, 255, 0)
    kWhite = (255, 255, 255)
    kBlack = (0, 0, 0)
    kOrange = (0, 165, 255)
    kPink = (203, 192, 255)
    kBrown = (42, 42, 165)
    kPurple = (128, 0, 128)
    kGray = (128, 128, 128)
    kSilver = (192, 192, 192)
    kGold = (0, 215, 255)
    kLime = (0, 255, 0)
    kNavy = (128, 0, 0)
    kTeal = (128, 128, 0)
    # 浅色系列
    kLightRed = (0, 128, 255)
    kLightGreen = (128, 255, 0)
    kLightBlue = (255, 128, 0)
    kLightYellow = (128, 255, 255)
    kLightMagenta = (255, 128, 255)
    kLightCyan = (255, 255, 128)
    # 深色系列
    kDarkRed = (0, 0, 128)
    kDarkGreen = (0, 128, 0)
    kDarkBlue = (128, 0, 0)
    kDarkYellow = (0, 128, 128)
    kDarkMagenta = (128, 0, 128)
    kDarkCyan = (128, 128, 0)
    # 其他常见颜色


def color2bgr(color):
    if isinstance(color, Enum):
        return color.value
    try:
        return Color[color].value
    except KeyError:
        print(f"未找到颜色 {color}，请检查颜色名称是否正确。")
        return None

def transform_points(pose, points):
    original_shape = points.shape
    ## 1. reshape
    points_reshaped = points.reshape(-1, 3)
    ## 2. rotation
    R = pose[:3, :3]
    points_r = np.einsum('ij,mj->mi', R, points_reshaped)
    ## 3. translation
    t = pose[:3, 3]
    points_tf = points_r + t
    return points_tf.reshape(original_shape)


def project_points(camk, points_camera):
    ## 1. filter points
    points_shape = points_camera.shape
    points_camera = points_camera.reshape(-1, 3)
    invalid_mask = points_camera[:, 2] <= 0
    points_camera[invalid_mask] = np.nan
    ## 2. unit
    points_camera = points_camera / points_camera[:, 2:3]
    pixels_image = np.einsum('ij,mj->mi', camk, points_camera)
    pixels_image = pixels_image.reshape(points_shape)[..., :2]
    return pixels_image

def filter_pixels(image_shape,pixels_image):
    ## 1. reshape
    h, w, _ = image_shape
    pixels_image = pixels_image.reshape(-1, 2)
    ## 2. filter nan
    valid_mask = ~np.isnan(pixels_image).any(axis=-1)
    valid_pixels = pixels_image[valid_mask]
    ## 3. filter out image
    valid_pixels = valid_pixels.astype(int)
    valid_mask = (valid_pixels[:, 0] >= 0) & (valid_pixels[:, 0] < w) & \
                 (valid_pixels[:, 1] >= 0) & (valid_pixels[:, 1] < h)
    valid_pixels = valid_pixels[valid_mask]
    return valid_pixels

def l2q(l):
    q = quaternion.from_float_array(l)
    return q

def l2p(l):
    return np.array(l)

def q2R(q):
    q = Quaternion(q)
    return q.rotation_matrix

def R2q(R):
    R = np.array(R)
    return Quaternion(matrix=R)

def pq2tf(pq):
    p,q = pq
    p = np.array(p)
    q = Quaternion(q)

    matrix4d = np.eye(4)
    matrix4d[:3, :3] = q2R(q)
    matrix4d[:3, 3] = p
    return matrix4d

def tf2pq(tf):
    tf = np.array(tf)
    p = tf[:3, 3]
    R = tf[:3, :3]
    q = R2q(R)
    return (p,q)

def pr2tf(pr):
    p,r = pr
    p = np.array(p)
    r = np.array(r)
    q = R2q(r)
    return pq2tf((p,q))

def tf2pr(tf):
    p,q = tf2pq(tf)
    p = np.array(p)
    q = Quaternion(q)
    r  = q2R(q)
    return (p,r)


## 1.p: vec3,q: Quan s: wlh/yxz
def s2oxyz(s: np.array, ratio=1.0):
    o = np.array([0, 0, 0])
    xl = s[1] * ratio
    yw = s[0] * ratio
    zh = s[2] * ratio
    x = np.array([xl, 0, 0])
    y = np.array([0, yw, 0])
    z = np.array([0, 0, zh])
    return np.array([o, x, y, z])

def pqs2oxyz(p : np.array,q: Quaternion,s: np.array,ratio = 1.0):
    oxyz = s2oxyz(s,ratio)
    o = q.rotation_matrix @ oxyz[0] + p
    x = q.rotation_matrix @ oxyz[1] + p
    y = q.rotation_matrix @ oxyz[2] + p
    z = q.rotation_matrix @ oxyz[3] + p
    return np.array([o, x, y, z])

if __name__ == '__main__':
    # 创建一个空白图像
    image_height = len(Color) * 50
    image_width = 200
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # 遍历每种颜色并在图像上绘制矩形
    for i, color in enumerate(Color):
        y_start = i * 50
        y_end = (i + 1) * 50
        cv2.rectangle(image, (0, y_start), (image_width, y_end), color.value, -1)
        # 在矩形旁边添加颜色名称
        cv2.putText(image, color.name, (10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 显示图像
    cv2.imshow('Color Chart', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()