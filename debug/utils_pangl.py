import pangolin
import OpenGL.GL as gl
from OpenGL.raw.GL.VERSION.GL_1_0 import glLineWidth

from utils import *

def set_gl_color(color_type: Color):
    bgr = color2bgr(color_type)
    gl.glColor3f(bgr[2], bgr[1], bgr[0])


def draw_grid(resolution=1., center=np.array([0, 0, 0])):
    num_cells = 250

    gl.glLineWidth(2)
    set_gl_color(Color.kBlack)

    row_starts = []
    row_ends = []
    col_starts = []
    col_ends = []
    for idx in range(-num_cells, num_cells + 1):
        delta = idx * resolution
        row_starts.append(np.array([delta, -num_cells * resolution, 0]) + center)
        row_ends.append(np.array([delta, num_cells * resolution, 0]) + center)

        col_starts.append(np.array([-num_cells * resolution, delta, 0]) + center)
        col_ends.append(np.array([num_cells * resolution, delta, 0]) + center)

    pangolin.DrawLines(row_starts, row_ends)
    pangolin.DrawLines(col_starts, col_ends)

def draw_pose(tf_world_key, length):
    o = transform_points(tf_world_key, np.array([0, 0, 0]))
    x = transform_points(tf_world_key, np.array([length, 0, 0]))
    y = transform_points(tf_world_key, np.array([0, length, 0]))
    z = transform_points(tf_world_key, np.array([0, 0, length]))
    set_gl_color(Color.kRed)
    pangolin.DrawLine([o, x])
    set_gl_color(Color.kGreen)
    pangolin.DrawLine([o, y])
    set_gl_color(Color.kBlue)
    pangolin.DrawLine([o, z])


def draw_arrow(start, end, line_width=2, color=Color.kBlack):
    set_gl_color(color)
    glLineWidth(line_width)
    ## 1.line1
    pangolin.DrawLine([start, end])

    arrow_length = np.linalg.norm(np.array(end) - np.array(start))
    arrow_head_length = 0.2 * arrow_length  # 箭头头部长度
    arrow_head_angle = np.pi / 6  # 箭头头部角度

    direction = np.array(end) - np.array(start)
    direction = direction / np.linalg.norm(direction)
    perpendicular = np.array([-direction[1], direction[0], 0])

    arrow_head_point1 = np.array(end) - arrow_head_length * (np.cos(arrow_head_angle) * direction + np.sin(arrow_head_angle) * perpendicular)
    arrow_head_point2 = np.array(end) - arrow_head_length * (np.cos(arrow_head_angle) * direction - np.sin(arrow_head_angle) * perpendicular)

    ## 2.line2/3
    pangolin.DrawLine([arrow_head_point1, end])
    pangolin.DrawLine([arrow_head_point2, end])

def corners_to_edges(corners):
    edges = [
        # bottom
        (0, 1), (1, 2), (2, 3), (3, 0),
        # top
        (4, 5), (5, 6), (6, 7), (7, 4),
        # edge
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    start_points = []
    end_points = []
    for start, end in edges:
        start_points.append(corners[start])
        end_points.append(corners[end])
    return np.array(start_points), np.array(end_points)

def TopView(center=np.array([0, 0, 0])):
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(1920, 1080, 2000, 2000, 960, 540, 0.1, 200),
        pangolin.ModelViewLookAt(center[0], center[1], center[2] + 70, center[0], center[1], center[2], 1, 0, 0))
    return scam
