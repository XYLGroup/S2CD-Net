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
        gttt = img(up:down, left:right, :);

        % ===== 6 Geometric Transformations =====
        augList = { ...
            @(x) x, ...                          % 0° Original Image
            @(x) imrotate(x, 90, 'bilinear', 'crop'), ...
            @(x) imrotate(x, 180, 'bilinear', 'crop'), ...
            @(x) imrotate(x, 270, 'bilinear', 'crop'), ...
            @(x) x(:,end:-1:1,:), ...            % Horizontal Flip
            @(x) x(end:-1:1,:,:)  ...            % Vertical Flip
            };

        for augIdx = 1:numel(augList)
            gtAug = augList{augIdx}(gttt);        % Apply augIdx transformation

            % Blur + Downsample
            sigma = 0.2 + (4 - 0.2)*rand;
            gaussKernel = fspecial('gaussian', 19, sigma);
            imgBlur = zeros(size(gtAug), 'single');
            for b = 1:size(gtAug,3)
                imgBlur(:,:,b) = imfilter(gtAug(:,:,b), gaussKernel, 'conv', 'replicate');
            end

            lq = single(imresize(imgBlur, factor));
            bic = single(imresize(lq, 1/factor));
            gt = single(gtAug);
            scale = round(1/factor);

            file_path = strcat('/home/descfly/zly/datasets/Pavia/mcodes/dataset/Pavia_x', num2str(scale), '/train/block_', ...
                               file_name, '_', num2str(index), '_aug', num2str(augIdx), '.mat');
            save(file_path, 'gt', 'lq', 'bic', '-v6');
        end
        index = index + 1;          % Count only incremented by 1, but 6 patches produced
    end
end
out = total_num;
end

        