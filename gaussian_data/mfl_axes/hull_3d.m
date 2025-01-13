%% load hull
clear
close all
clc

exp = '2022_03_03'
path = 'H:\My Drive\dark 2022\2022_03_03\hull\hull_Reorder\'
easyWand_name = '3+4_post_03_03_2022_skip5_easyWandData.mat'


% path = 'H:\My Drive\dark 2022\2022_05_19\hull\hull_Reorder\'
% easyWand_name = 'wand_data1_19_05_2022_skip5_easyWandData'

movie = 19
mov_name = sprintf('mov%d',movie)
struct_file_name = sprintf('\\Shull_mov%d',movie)
load([path,mov_name,'\hull_op\',struct_file_name])

hull3d_file_name = sprintf('\\hull3d_mov%d',movie)
load([path,mov_name,'\hull_op\',hull3d_file_name])

% save_dir = sprintf('G:/My Drive/%s/',exp)
% save_camera_matrices = [save_dir,'camera'];
% save_3d_hull = [save_dir,'3d_data/'];
% save_2d_data = [save_dir,'2d_data/'];

% mkdir([save_dir])
% mkdir(save_camera_matrices)
% mkdir(save_3d_hull)
% mkdir(save_2d_data)

% load sparse

for cam = 1:1:4
    sparse_file = sprintf('\\mov%d_cam%d_sparse.mat',movie,cam)
    sp{cam} = load([path,mov_name,sparse_file])
end

%%

frame_sparse = 500;
frame = find(Shull.frames == frame_sparse);
body = hull3d.body.body4plot{frame};
wing_left = hull3d.leftwing.hull.hull3d{frame};
wing_right = hull3d.rightwing.hull.hull3d{frame};


real_coords = Shull.real_coord{frame}
body_3d = [real_coords{1}(body(:,1))',real_coords{2}(body(:,2))',real_coords{3}(body(:,3))']
wing_left_3d = [real_coords{1}(wing_left(:,1))',real_coords{2}(wing_left(:,2))',real_coords{3}(wing_left(:,3))']
wing_right_3d = [real_coords{1}(wing_right(:,1))',real_coords{2}(wing_right(:,2))',real_coords{3}(wing_right(:,3))']
ew2lab = Shull.rotmat_EWtoL;
fly = [body_3d;wing_left_3d;wing_right_3d];
% writematrix(fly,[save_3d_hull,'fly']);
%% world axes - from easywand
figure;
load([path,easyWand_name])


subplot(1,2,1)
plot_camera(easyWandData.rotationMatrices,easyWandData.DLTtranslationVector,[1,0,0;0,1,0;0,0,1],'Easy wand')
subplot(1,2,2)

plot_camera(easyWandData.rotationMatrices,easyWandData.DLTtranslationVector,ew2lab,'Lab')
%% world axes - from coefs
figure;
load([path,easyWand_name])

for j= 1:1:4

[R,K,X0,H] = decompose_dlt(easyWandData.coefs(:,j),easyWandData.rotationMatrices(:,:,j)');
rotation(:,:,j) = R./vecnorm(R')'; 
translation(:,:,j) = X0; 
k_all(:,:,j) = K;
pmdlt{j} = [K*R,-K*R*X0];
end
plot_camera(rotation,translation,[1,0,0;0,1,0;0,0,1],'standard wand')

%%
cam = 4
im = ImfromSp([800,1280],sp{cam}.frames(frame_sparse).indIm);

pt2d = pmdlt{cam}*fly_h';
pt2d =( pt2d./pt2d(3,:))';
figure;
imshow(im);hold on
scatter(pt2d(:,1),pt2d(:,2),'r.')


%% project from 3d to 2d
figure;
d = 10
for cam = 1:1:4
fly_h = [fly,ones(size(fly,1),1)];
pt2d = pmdlt{cam}*fly_h';
pt2d =( pt2d./pt2d(3,:))';

image_data = sp{cam}.frames(frame_sparse).indIm;
if cam == 1
    image_data(:,1) = 801 - image_data(:,1);
end
crop = double([min(image_data(:,2)) - d, min(image_data(:,1)) - d,max(image_data(:,2)) + d,max(image_data(:,1)) + d]);
im = ImfromSp([800,1280],sp{cam}.frames(frame_sparse).indIm);

subplot(2,2,cam)
[uv] = dlt_inverse(easyWandData.coefs(:,cam),fly);
imshow(im(crop(2):crop(4),crop(1):crop(3)));hold on
scatter(uv(:,1) - crop(1)+1,801-uv(:,2) - crop(2)+1);hold on
scatter(pt2d(:,1) - crop(1)+1,pt2d(:,2) - crop(2)+1,'r.')
end

legend('ew','standard')

%% world axes - from coefs
figure;
load([path,easyWand_name])
clr = {'r','g','b'}
for j = 1:1:4
subplot(2,2,j)
[R,K,X0,H] = decompose_dlt(easyWandData.coefs(:,j),easyWandData.rotationMatrices(:,:,j)');
t = R*X0;
fly_cam = (R*fly' + t)';
scatter3(fly_cam(:,1),fly_cam(:,2),fly_cam(:,3),'.');hold on
for k = 1:1:3
quiver3(t(1),t(2),t(3),R(1,k),R(2,k),R(3,k),0.005,clr{k});axis equal
end
view(0,90)
ttl = sprintf('cam%d',j)
xlabel('x');ylabel('y');zlabel('z')
title(ttl)
end
%%
j = 2
[R,K,X0,H] = decompose_dlt(easyWandData.coefs(:,j),easyWandData.rotationMatrices(:,:,j)');

znear = 1
zfar = 100
fovX = K(1,1)
fovY =K(2,2)

P = getProjectionMatrix(znear, zfar, fovX, fovY)

proj_mat = [R,R*X0]*P

pt2d = (proj_mat*fly_h')
pt2d = pmdlt{cam}*fly_h';
pt2d =( pt2d./pt2d(3,:))';

figure;
cam = 2
im = ImfromSp([800,1280],sp{cam}.frames(frame_sparse).indIm);
imshow(im);hold on
scatter(pt2d(:,1),801-pt2d(:,2))