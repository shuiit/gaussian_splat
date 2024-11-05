function save_images(sp,save_path)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
frameSize = [800,1280]

for cam = 1:1:4
for frame = 1:1:500%length(sp{cam}.frames)
     indim = sp{cam}.frames(frame).indIm;

    if cam == 1
        indim(:,1) = 801 - sp{cam}.frames(frame).indIm(:,1);

    end

    im_name = sprintf('P%dCAM%d.jpg',frame,cam);
    [Im] = ImfromSp(frameSize,indim);
    im = im2gray(double(Im/255/255));
    imwrite(im,[save_path,im_name]);
end
end
end