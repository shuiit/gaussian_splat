%% load hull
clear
close all
clc

exp = '2022_03_03'
path = 'H:\My Drive\dark 2022\2022_03_03\hull\hull_Reorder\'
easyWand_name = '3+4_post_03_03_2022_skip5_easyWandData.mat'


movie = 19
mov_name = sprintf('mov%d',movie)
struct_file_name = sprintf('\\Shull_mov%d',movie)
load([path,mov_name,'\hull_op\',struct_file_name])

hull3d_file_name = sprintf('\\hull3d_mov%d',movie)
load([path,mov_name,'\hull_op\',hull3d_file_name])

load([path,easyWand_name])


% load sparse
for cam = 1:1:4
    sparse_file = sprintf('\\mov%d_cam%d_sparse.mat',movie,cam)
    sp{cam} = load([path,mov_name,sparse_file])
end

%%
path = 'G:\My Drive\Research\gs_data\'
save_path = [path,mov_name,'_',exp,'\','3d_pts','\']
mkdir(save_path)

hull_mat_file(hull3d.body.body4plot,[save_path,'body.mat'],hull3d.frames);

hull_mat_file(hull3d.rightwing.hull.hull3d,[save_path,'rwing.mat'],hull3d.frames);
hull_mat_file(hull3d.leftwing.hull.hull3d,[save_path,'lwing.mat'],hull3d.frames);

real_coord(Shull,[save_path,'real_coord.mat'])
%%
path = 'G:\My Drive\Research\gs_data\'
path = 'D:\Documents\'

save_path = [path,mov_name,'_',exp,'\','images','\']
mkdir(save_path)
save_images(sp,save_path)


%% world axes - from coefs
path = 'G:\My Drive\Research\gs_data\'
frame_sparse = 450

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
fly_h = [fly,ones(size(fly,1),1)];

for j= 1:1:4
save_path = [path,mov_name,'_',exp,'\','camera_KRX0']
[R,K,X0,H] = decompose_dlt(easyWandData.coefs(:,j),easyWandData.rotationMatrices(:,:,j)');
camera(:,:,j) = [K,R,X0];
rotation(:,:,j) = R; 
translation(:,:,j) = X0; 
pmdlt{j} = [K*R,-K*R*X0];

end
plot_camera(rotation,translation,[1,0,0;0,1,0;0,0,1],'standard wand')
save(save_path,'camera');
cam = 2
im = ImfromSp([800,1280],sp{cam}.frames(frame_sparse).indIm);

pt2d = pmdlt{cam}*fly_h';
pt2d =( pt2d./pt2d(3,:))';
figure;
imshow(im2gray(im/255/255));hold on
scatter(pt2d(:,1),801-pt2d(:,2),'r.')

