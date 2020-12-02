import numpy as np
import matplotlib.pyplot as plt
import cv2

np.random.seed(1)

def generate_collages(  
        textures,
        batch_size,
        segmentation_regions=10,
        anchor_points=None):
    N_textures = textures.shape[0]
    img_size= textures.shape[1]
    masks, gt_masks = generate_random_masks(img_size, batch_size, segmentation_regions, anchor_points)
    textures_idx = [np.random.randint(0, N_textures, size=batch_size) for _ in range(segmentation_regions)]
    gt_textures_idx = textures_idx[0][:]
    gt_textures = []
    tmp = []
    for tex in gt_textures_idx:
        tmp = cv2.resize(textures[tex], (64, 64))
        gt_textures.append(tmp)
    batch = sum(textures[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    return batch, gt_masks, np.stack(gt_textures) # query, gt, ref

def generate_random_masks(img_size=256, batch_size=1, segmentation_regions=10, points=None):
    xs, ys = np.meshgrid(np.arange(0, img_size), np.arange(0, img_size))

    if points is None:
        n_points = np.random.randint(2, segmentation_regions + 1, size=batch_size)
        points   = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]
        
    masks = []
    ground_truth = []
    for b in range(batch_size):
        dists_b = [np.sqrt((xs - p[0])**2 + (ys - p[1])**2) for p in points[b]]
        voronoi = np.argmin(dists_b, axis=0)
        masks_b = np.zeros((img_size, img_size, segmentation_regions))
        for m in range(segmentation_regions):
            masks_b[:,:,m][voronoi == m] = 1
        masks.append(masks_b)
    stack_masks = np.stack(masks)
    gt_masks = stack_masks[:, :, :, 0] # pick the first texture as the gt
    return np.stack(masks), np.stack(gt_masks)

def generate_collages(N=2000): # size of the dataset
    textures = np.load('train_texture.npy')
    collages, gt_masks, gt_textures = generate_collages(textures, batch_size=N)
    np.save('train_query_2000.npy', collages)
    np.save('train_gt_2000.npy', gt_masks)
    np.save('train_ref_2000.npy', gt_textures)
    
if __name__ == '__main__':
    generate_collages()