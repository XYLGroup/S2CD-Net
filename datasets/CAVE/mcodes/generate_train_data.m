clc
clear
close all
rng(10, 'twister');

% List all '.mat' file in folder
file_folder=fullfile('/home/descfly/zly/datasets/CAVE/mcodes/train');
file_list=dir(fullfile(file_folder,'*.mat'));
file_names={file_list.name};

% store cropped images in folders
for i = 1:1:numel(file_names)
    name = file_names{i};
    name = name(1:end-4);
    load(strcat('/home/descfly/zly/datasets/CAVE/mcodes/train/',file_names{i}));
    crop_image(gt, 64, 32, 1/4, name);
end
