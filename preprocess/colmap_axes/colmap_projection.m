
pth = 'D:\Documents\south-building\sparse\'
pth_im = 'D:\Documents\south-building\images\'

filename_images = [pth, 'images.txt']
filename_3d_pts = [pth, 'points3D.txt']
image_id = 21


pt3d_tbl = get_3d_pts(filename_3d_pts,image_id);
[images_tbl,pts_2d] = get_quat(filename_images,image_id);

image_row = images_tbl(find(images_tbl.IMAGE_ID == image_id),:)
%%
[c,ia,ib] = intersect(pt3d_tbl.POINT3D_ID,pts_2d{1})
for i = 1:1:length(ib)
    pts(i,1:2) = pts_2d{1}(ib(i) - 2 : ib(i) - 1)';
end
figure
im = imread([pth_im,image_row.NAME{1}]);
imshow(im);hold on
scatter(pts(:,1),pts(:,2))

%%
figure;
ax1 = subplot(2,1,1)
xyz = [pt3d_tbl.X,pt3d_tbl.Y,pt3d_tbl.Z];
R1 = qvec2rotmat(table2array(image_row(1,2:5))); % rotation matrix, world to camera, each column is an axes of the camera in world coordinates
t1 = table2array(image_row(1,6:8)); % translation vector, the origin of the world FoR in camera axes

clr = {'r','g','b','m'}
plot3(pt3d_tbl.X,pt3d_tbl.Y,pt3d_tbl.Z,'.');xlabel('x');ylabel('y');zlabel('z');axis equal;hold on % plot the 3d points in world coordinates.
R = R1'; % transpose the rotation matrix. R is cam to world
t = -R*t1'; % rotate the translation vector from cam to world
for k = 1:1:3 % plot the camera axes in world coordinates
quiver3(t(1),t(2),t(3),R(1,k),R(2,k),R(3,k),color = clr{k});hold on
end
axis equal
ax2 = subplot(2,1,2)

% plot 3d points in camera coordinates. plot the world coordinates as
% presived by the camera (X - right, Y - down , Z - inside)
campts = (R1*xyz' + t1')'; % 3d points - world to cam
plot3(campts(:,1),campts(:,2),campts(:,3),'.');axis equal;hold on

% plot the location of the origin and direction of the axes of the world
% (as seen by the camera)
for k = 1:1:3 
quiver3(t1(1),t1(2),t1(3),R1(1,k),R1(2,k),R1(3,k),color = clr{k});hold on
end
linkaxes([ax1,ax2],'xyz')





%% plot all cameras in world FoR
figure

for row = 1:1:100
[R] = qvec2rotmat(table2array(images_tbl(row,2:5))); % world 2 cam
t = table2array(images_tbl(row,6:8)); % cam?
R = R'; % cam to world
t = -R*t'; % 
for k = 1:1:3
quiver3(t(1),t(2),t(3),R(1,k),R(2,k),R(3,k),color = clr{k});hold on
hold on
end
axis equal
plot3(0,0,0,'.')

end
xyz = [pt3d_tbl.X,pt3d_tbl.Y,pt3d_tbl.Z];
plot3(pt3d_tbl.X,pt3d_tbl.Y,pt3d_tbl.Z,'.');xlabel('x');ylabel('y');zlabel('z');axis equal

%% Project 3d points to 2d image

xyz = [pt3d_tbl.X, pt3d_tbl.Y, pt3d_tbl.Z,ones(size(pt3d_tbl.Z,1),1)]; 
K = [2559.68 0 1536;
     0 2559.68 1152;
     0 0 1];
R1 = qvec2rotmat(table2array(image_row(1,2:5))); % rotation matrix, world to camera, each column is an axes of the camera in world coordinates
t1 = table2array(image_row(1,6:8)); % translation vector, the origin of the world FoR in camera axes

p = [R1,t1'];
pm = K*p;
pm = pm%/pm(3,4);
pt2d = pm*xyz';




 [R,K,X0] = decompose_dlt(pm)
figure;
pt2d =( pt2d./pt2d(3,:))'
imshow(im);hold on
scatter(pts(:,1),pts(:,2));hold on
scatter(pt2d(:,1),pt2d(:,2),'r.')


[K*R,-K*R*X0]
x0real = -R1'*t1';
%%
xyz = [pt3d_tbl.X, pt3d_tbl.Y, pt3d_tbl.Z]; 


znear = 1
zfar = 100000000

% World transformation matrix (R1 and t1)
world = [R1, t1'; 0, 0, 0, 1];  % Ensure homogeneous transformation matrix is 4x4

% Projection matrix (assuming getProjectionMatrix is defined elsewhere)
P = getProjectionMatrix(znear, zfar, fovX, fovY);  % Projection matrix

% Ensure xyz has homogeneous coordinates (4xN matrix)
xyz_h = [xyz, ones(size(xyz, 1), 1)];  % Add 1 as the fourth component to each 3D point

% Perform world-to-view transformation followed by projection
dpt = world * xyz_h';  % Apply world transformation (4x4 matrix multiplication)
dpt = (dpt ./ dpt(4, :))';  % Perspective divide by the 4th row (normalize homogeneous coordinates)

% Convert from normalized device coordinates (NDC) to pixel coordinates
% Assuming image size is 1280x800
imsz = size(im);
x = (dpt(:, 1) + 1) * (imsz(2) / 2);  % NDC x from [-1, 1] to [0, 1280]
y = (1 - dpt(:, 2)) * (imsz(1) / 2);   % NDC y from [-1, 1] to [0, 800] (invert axis for image space)

% Plot 3D points after transformation (optional)
figure;
plot3(dpt(:, 1), dpt(:, 2), dpt(:, 3), '.');

% Plot the points on the image
figure; imshow(im); hold on;
scatter(pts(:,1),pts(:,2));hold on

scatter(x, y,'.r');
