function convGray2RGB()
% The function takes input dir, reads grayscale files and writes RGB files
% in output dir
% The images are only stacked, no color transformation is used

    in_dir1 = '/home/himanshu/Anjali/data/grayData/HC';
    in_dir2 = '/home/himanshu/Anjali/data/grayData/IC';
    out_dir1 = '/home/himanshu/Anjali/data/rgbData/HC';
    out_dir2 = '/home/himanshu/Anjali/data/rgbData/IC';
    
    cgr(in_dir1, out_dir1)
    disp('HC done');
    cgr(in_dir2, out_dir2)
    disp('IC done');
end

function cgr(in_dir, out_dir)
    files = dir2(in_dir);
    for i=1:size(files,1)
        im = imread( fullfile(in_dir,files(i).name) );
        assert (size(im,3) == 1);
        imout = zeros(size(im,1), size(im,1), 3);
        imout = uint8(imout);
        imout(:,:,1) = im;
        imout(:,:,2) = im;
        imout(:,:,3) = im;
        
        [~,name,~] = fileparts( fullfile(in_dir,files(i).name) );
        imwrite(imout, fullfile(out_dir, strcat(name, '.png')));
    end
end