from statistic import *

if __name__ == "__main__":
    points_dir = r"F:\Video\behavioral_experiment\jsm20240730\resized\result_ym\rectify\rectify_WIN_20240728_09_39_10_Pro.csv"
    video_path = 'F:\\Video\\behavioral_experiment\\jsm20240730\\resized\\resized_ym\\WIN_20240728_09_39_10_Pro.avi'
    transformed_points = np.genfromtxt(points_dir, delimiter=',')
    rect = transformed_points[0:4, :]
    path = transformed_points[4:, :]
    mouse = YMaze(path=path, video=video_path, width=np.max(rect), height=np.max(rect),
                  down_arm_array=rect, verbose=True)
    print(mouse.standstill_count())