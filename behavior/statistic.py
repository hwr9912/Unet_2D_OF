import numpy as np
from tqdm import tqdm
import cv2
import re


class _BehaviorTest:
    def __init__(self,
                 path: np.array,
                 video: str,
                 radius: int = 10,
                 interval_time: int = 1,
                 verbose=False):
        """
        :param path: N*2的数组
        :param video: 识别后掩膜视频地址
        :param radius: 动物静止判定半径阈值
        :param interval: 动物静止判定时间阈值，单位为秒
        :param verbose: 繁复模式
        """
        # 运动路径点
        if path.shape[1] != 2:
            raise ValueError("Shape of input should be (N, 2)!")
        self.path = path
        self.length = path.shape[0]
        # 掩膜视频位置
        self.video = video
        # 提取视频帧数信息
        try:
            capture = cv2.VideoCapture(self.video)
            self.fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
        except:
            RuntimeError("Please check video path: input video path cannot be read!")
        # 计算相邻点之间的距离
        self.distances = np.sqrt(np.sum(np.diff(self.path, axis=0) ** 2, axis=1))
        # 繁复模式
        self.verbose = verbose
        if self.verbose:
            print(f"Total {self.length} frames with {self.fps} frame per sec.")
        # 初始化静止判定条件
        self.radius = radius
        self.interval = interval_time * self.fps
        # 初始化运动状态数据
        self.status = self.standstill_status()

    def total_distance(self):
        # 计算总距离
        return np.sum(self.distances)

    def average_speed(self):
        # 计算总距离
        total_distance = self.total_distance()
        # 计算总时间 (每行代表 1/30 秒)
        total_time = len(self.path) / self.fps
        # 计算平均速度
        average_speed = total_distance / total_time
        return average_speed

    def standstill_status(self):
        """
        寻找静止点
        要求寻找所有符合要求的圆（给定半径），要求为圆内包含尽可能能多次序相连的点，且点的数目超过指定的阈值，输出所有圆圆心的位置
        """
        status = np.repeat('M', repeats=self.path.shape[0])
        # 如果相邻点之间的距离小于直径，则可能符合提取前一个点索引
        standstill = np.where(self.distances < 2 * self.radius)[0]
        # 繁复模式
        if self.verbose:
            standstill = tqdm(standstill)
        # 逐索引遍历
        end = 0
        for i in standstill:
            # 忽略掉已经确定为静止的点：
            #   如果位置小于等于上一个终止点索引+1，则跳过
            if end <= i:
                # 所有数据归零
                points = self.path[i:] - np.average(self.path[i:], axis=0)
                while True:
                    # 计算距离矩阵
                    distances = np.sum(np.diff(points, axis=0) ** 2, axis=1)
                    # 找出最大值索引
                    cutpoint = np.argmax(distances)
                    # 如果帧数间隔小于阈值则退出
                    if cutpoint < self.interval:
                        end = i + cutpoint
                        break
                    # 如果最大值大于半径，则仅保留cutpoint之前的数据，再次计算
                    if distances[cutpoint] > self.radius ** 2:
                        points = points[0:cutpoint, ]
                    # 如果最大值小于半径，则判定其间的所有状态均为静止并退出循环
                    else:
                        # 终结点设置为目前的points长度+i
                        end = i + points.shape[0]
                        status[i:end] = 'S'
                        break
            else:
                continue
        return status

    def standstill_count(self):
        return len(re.findall(r'S+', ''.join(self.status)))

    def standstill_time(self):
        return np.count_nonzero(self.status == 'S') / self.fps


class OpenField(_BehaviorTest):
    def __init__(self,
                 path, video, border_array,
                 lower_ratio=0.25, upper_ratio=0.75, radius=10, interval_time=1,
                 verbose=False):
        """
        旷场实验模式类
        :param path: 路径矩阵
        :param video: 视频地址
        :param border_array_path: 边界矩阵
        :param lower_ratio: 中央区尺寸低点占比
        :param upper_ratio: 中央区尺寸高点占比
        :param radius: 静止识别半径阈值
        :param interval_time: 静止识别时间阈值
        :param verbose:繁复模式
        """
        super().__init__(path=path, video=video, radius=radius, interval_time=interval_time,
                         verbose=verbose)
        # 未矫正前迷宫四角位置
        self.border_array = border_array.astype(np.int32)
        # 核心占比
        self.core_region_ratio = (lower_ratio, upper_ratio)
        # 跨边界状态
        self.location = []
        self._location()

    def _location(self):
        # 逐帧读取视频
        cap = cv2.VideoCapture(self.video)
        # 设置捕获视频为灰度模式
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('8', 'U', 'C', '1'))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # 初始化区域标识矩阵
        region = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.uint8)  # 创建一个空白图像
        # 定义四边形的顶点
        # 注意：顶点需要按顺序（顺时针或逆时针）提供
        center_point = np.average(self.border_array, axis=0)
        pts = np.int32((self.border_array - center_point) / 2 + center_point)
        # 绘制实心四边形
        # 参数1：图像
        # 参数2：顶点数组, np.int32
        # 参数3：颜色（BGR格式）
        cv2.fillPoly(region, [pts], color=1)
        # 显示图像
        # cv2.imshow('Solid Quadrilateral', region * 255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 中心区为2，外围为1
        region += 1
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = np.int8(frame > 128) + region
                if 2 in result:
                    if 3 in result:
                        self.location.append('B')  # border
                    else:
                        self.location.append('O')  # out
                else:
                    self.location.append('I')  # in
        else:
            RuntimeError("Please check video path: input video path cannot be read!")
        cap.release()
        self.location = np.array(self.location)

    def center_staying_count(self):
        """
        :return: 中央区停留次数
        """
        return len(re.findall(r"I+", ''.join(self.location)))

    def center_staying_time(self):
        """
        :return: 中央区停留时间，单位sec
        """
        time = np.count_nonzero(self.location == "I") / self.fps
        return time

    def center_staying_time_ratio(self):
        """
        :return: 中央区停留时间占比
        """
        return self.center_staying_time() * self.fps / self.length

    def outside_staying_count(self):
        """
        :return: 中央区停留次数
        """
        return len(re.findall(r"O+", ''.join(self.location)))

    def outside_staying_time(self):
        """
        :return: 中央区停留时间，单位sec
        """
        time = np.count_nonzero(self.location == "O") / self.fps
        return time

    def outside_staying_time_ratio(self):
        """
        :return: 中央区停留时间占比
        """
        return self.outside_staying_time() * self.fps / self.length

    def border_staying_count(self):
        """
        :return: 中央区停留次数
        """
        return len(re.findall(r"B+", ''.join(self.location)))

    def border_staying_time(self):
        """
        :return: 中央区停留时间，单位sec
        """
        return np.count_nonzero(self.location == "B") / self.fps

    def border_staying_time_ratio(self):
        """
        :return: 中央区停留时间占比
        """
        return self.border_staying_time() * self.fps / self.length


class NewObjectRecognition:
    def __init__(self, path, width, height, down_arm_array):
        super().__init__(path, width, height)
        if down_arm_array.shape != (4, 2):
            raise ValueError("Shape of down_arm_array should be (4, 2)!")
        self.width = width
        self.height = height
        self.down_arm_region = down_arm_array
        self.arm_location = None


class YMaze(_BehaviorTest):
    def __init__(self, path, video, width, height, down_arm_array,
                 radius: int = 10,
                 interval_time: int = 1,
                 verbose=False):
        super().__init__(path=path, video=video, radius=radius, interval_time=interval_time, verbose=verbose)
        if down_arm_array.shape != (4, 2):
            raise ValueError("Shape of down_arm_array should be (4, 2)!")
        self.down_arm_region = down_arm_array
        self.arm_location = self._spontaneous_alternation()

    def _spontaneous_alternation(self):
        """
        轮替计数，根据x轴位置识别每一帧小鼠的位置，返回所在臂
        :return: 返回轮替计数次数
        """
        # 确定分划点
        threshold_M = np.min(self.down_arm_region[:, 1])
        threshold_L = np.min(self.down_arm_region[:, 0])
        threshold_R = np.max(self.down_arm_region[:, 0])
        # 构建字符数组 "DDDDDDDDDDDDDDD"
        location_str = np.repeat("D", repeats=self.path.shape[0])
        # 生成悬臂位置字符串 "LLDDRRRRMMMMDDDDDLL"
        location_str[self.path[:, 1] < threshold_M] = "M"
        location_str[self.path[:, 0] < threshold_L] = "L"
        location_str[self.path[:, 0] > threshold_R] = "R"
        return location_str

    def alter_cycle_count(self):
        """
        :return: 返回轮替次数，允许重叠
        """
        location_str = ''.join(self.arm_location)
        # 去掉重复出现的部分 "LDRDL"
        location_str = re.sub(r'([DLRM])\1+', r'\1', location_str)
        # 计数轮替字符串
        count = len(re.findall(r'(LDR|DLR)', location_str)) + len(re.findall(r'(LRD|RLD)', location_str)) + len(
            re.findall(r'(DRL|RDL)', location_str))
        return count


def order_points(pts: np.array):
    """
    初始化一个列表以将点按左上、右上、右下和左下的顺序排列
    :param pts: 四点矩阵
    :return: 排序后矩阵
    """
    rect = np.zeros((4, 2), dtype="float32")

    # 计算左上角和右下角的点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上角和左下角的点
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # 返回按顺序排列的点
    return rect


def four_point_transform(rect):
    """
    初始化矩形矫正的模型
    :param rect:输入四个点构成的矩形
    :return:
    """
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return M
