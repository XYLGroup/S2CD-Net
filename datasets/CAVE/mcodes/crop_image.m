function [out] = crop_image(img, patch_size, stride, factor, file_name)

img = double(img);

[H, W, C] = size(img);
p = patch_size;
pat_col_num = 1:stride:(H - p + 1);
pat_row_num = 1:stride:(W - p + 1);
total_num = length(pat_col_num) * length(pat_row_num);
index = 1;

% crop a single patch from whole image
for i=1:length(pat_col_num)
    for j = 1:length(pat_row_num)
        up = pat_col_num(i);
        down = up + p - 1;
        left = pat_row_num(j);
        right = left + p - 1;
        gt = img(up:down, left:right, :);

        sigma = 0.2 + (4 - 0.2)*rand;
        gaussKernel = fspecial('gaussian', 9, sigma);
        imgBlur = zeros(size(gt), 'single');
        for b = 1:size(gt,3)
            imgBlur(:,:,b) = imfilter(gt(:,:,b), gaussKernel, 'conv', 'replicate');
        end

        lq = single(imresize(imgBlur, factor));
        bic = single(imresize(lq, 1/factor));
        gt = single(gt);
        scale = round(1/factor);
        file_path = strcat('/home/descfly/zly/datasets/CAVE/mcodes/dataset/CAVE_x', num2str(scale), '/train/block_', file_name, '_', num2str(index), '.mat');
        save(file_path,'gt','lq','bic','-v6');
        index = index + 1;
    end
end
out = total_num;
end

        