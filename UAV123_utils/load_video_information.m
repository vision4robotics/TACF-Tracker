function seq = load_video_information(type)

switch type
    case 'UAV123_10fps'
        database_folder = 'seq\data_seq';
        ground_truth_folder = 'seq\anno';
        video_name = choose_video_UAV(ground_truth_folder);
        seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type);
        seq.video_name = video_name;
end