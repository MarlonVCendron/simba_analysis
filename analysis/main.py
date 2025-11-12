from analysis.load import read_raw_dlc
from analysis.utils.missing_sessions import print_session_summary, print_complete_rats
from analysis.visualization.plot_keypoint_diffs import plot_keypoint_diffs
from analysis.visualization.visualize_trajectory import visualize_all_trajectories

def main():
    dlc_data = read_raw_dlc()
    # print_session_summary(dlc_data)
    # print_complete_rats(dlc_data)
    # plot_keypoint_diffs(dlc_data)
    visualize_all_trajectories(dlc_data)

if __name__ == "__main__":
    main()

