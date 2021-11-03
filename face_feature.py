import numpy as np
import mediapipe as mp
import cv2
import math
import glob
from sklearn.cluster import DBSCAN
from descartes import PolygonPatch
from makeup import get_img_info, get_pt_ref, find_top2_clusters
from UV_utils import uv_coords, uv_map, keypts
from ColorConverter import BGR_to_HSV
from polygon_check import is_inside_polygon
from polygon_wrap import concaveHull, convexHull
from bezier import evaluate_bezier



def get_eyebrow(img_path):
    landmarks, idx_coords, img = get_img_info(img_path)
    # idx of all eyebrow keyoints
    # get keypoint idx of eyebrows
    left_eyebrow_keypts = keypts["leftEyebrowUpper"] + keypts["leftEyebrowLower"][::-1]
    right_eyebrow_keypts = keypts["rightEyebrowUpper"] + keypts["rightEyebrowLower"][::-1]
    eyebrow_keypts = left_eyebrow_keypts + right_eyebrow_keypts
    # get eyebrow coordinates
    eyebrow_coords = np.array([idx_coords[i] for i in eyebrow_keypts])
    # get eyebrow uvcoords
    left_eyebrow_uvs = [uv_coords[i] for i in left_eyebrow_keypts]
    right_eyebrow_uvs = [uv_coords[i] for i in right_eyebrow_keypts]

    # clip image of including eyebrow, padding for 5 px
    left, right, up, down = eyebrow_coords[:,0].min(), eyebrow_coords[:,0].max(), eyebrow_coords[:,1].min(), eyebrow_coords[:,1].max()
    clip = img[up-5 : down+5, left-5 : right+5, :]

    # edge detection
    edges = cv2.Canny(clip, 25, 60, L2gradient=True) #25,60
    cv2.imwrite("./CPM-Real_eyebrowTest/edge/"+img_path[20:], edges)
    
    # preserve edges in eyebrow region only
    found_eyebrow_uv_coords = []
    found_eyebrow_xy_coords = []

    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i,j]==255:
                coord = i+up-5, j+left-5
                normalized_xy = coord[1]/img.shape[1], coord[0]/img.shape[0]
                ratios, indices = get_pt_ref(normalized_xy, landmarks)
                if ratios!=[1,0,0]:
                    uv = np.matmul(ratios, [uv_coords[indices[0]], 
                                            uv_coords[indices[1]], 
                                            uv_coords[indices[2]]])
                    # check if it is within eyebrow region
                    if is_inside_polygon(points=left_eyebrow_uvs, p=uv) or is_inside_polygon(points=right_eyebrow_uvs, p=uv):
                        found_eyebrow_uv_coords.append(uv)
                        found_eyebrow_xy_coords.append((coord[1], coord[0]))
                        cv2.circle(img, (coord[1], coord[0]), 0, (0,0,255), 1)

                    else:
                        edges[i,j]=0

    # find biggest 2 clusters (further denoising)
    clustering = DBSCAN(eps=0.015).fit(found_eyebrow_uv_coords)
    idxA, idxB  = find_top2_clusters(clustering.labels_)
    eyebrowA = [found_eyebrow_uv_coords[i] for i in idxA]
    eyebrowB = [found_eyebrow_uv_coords[i] for i in idxB]

    #concave hull
    alpha=20
    polygonA = concaveHull(eyebrowA, alpha)
    polygonB = concaveHull(eyebrowB, alpha)

    def draw_eyebrow(polygon):
    
        # draw bounding polygon for first eyebrow
        patch = PolygonPatch(polygon, zorder=2)
        vertices = patch.get_path()._vertices
        
        coordinates = []

        # pre=vertices[0]
        # for i in vertices[1:]:
        #     start = pre
        #     ratios, indices = get_pt_ref(start, uv_coords)
        #     start_xy = np.matmul(ratios, [idx_coords[indices[0]], 
        #                                 idx_coords[indices[1]], 
        #                                 idx_coords[indices[2]]])
        #     start_xy[0] = min(math.floor(start_xy[0]), img.shape[1] - 1)
        #     start_xy[1] = min(math.floor(start_xy[1]), img.shape[0] - 1)


        #     end = i
        #     ratios, indices = get_pt_ref(end, uv_coords)
        #     end_xy = np.matmul(ratios, [idx_coords[indices[0]], 
        #                                 idx_coords[indices[1]], 
        #                                 idx_coords[indices[2]]])
        #     end_xy[0] = min(math.floor(end_xy[0]), img.shape[1] - 1)
        #     end_xy[1] = min(math.floor(end_xy[1]), img.shape[0] - 1)

        #     start_xy=start_xy.astype("int")
        #     end_xy=end_xy.astype("int")
        #     cv2.line(img, tuple(start_xy), tuple(end_xy), (255,0,0), 2)
        #     pre=i

        for i in vertices:
            ratios, indices = get_pt_ref(i, uv_coords)
            x,y = np.matmul(ratios, [idx_coords[indices[0]], 
                                        idx_coords[indices[1]], 
                                        idx_coords[indices[2]]])
            x = min(math.floor(x), img.shape[1] - 1)
            y = min(math.floor(y), img.shape[0] - 1)
            if [x,y] not in coordinates:
                coordinates.append([x,y])
  
        beizerpts = evaluate_bezier(np.array(coordinates), 20)
        beizerpts = np.append(beizerpts,[beizerpts[0]], axis=0)
        pre=beizerpts[0]
        
        for i in beizerpts[1:]:
            start = pre
            end = i
            if np.array_equal(start, end):
                pre=i
                continue
            cv2.line(img, tuple(start), tuple(end), (255,0,0), 1)
            pre=i


    


        
    draw_eyebrow(polygonA)
    draw_eyebrow(polygonB)

    #temp code: without polygon wrapping
    # for i in idxA+idxB:
    #     cv2.circle(img, found_eyebrow_xy_coords[i], 0, (255,0,0), 1)

    cv2.imwrite("./CPM-Real_eyebrowTest/label/"+img_path[20:], img)

    # return [eyebrowA[i].tolist() for i in polygonA], [eyebrowB[j].tolist() for j in polygonB]
    return


if __name__ == "__main__":

    # eyebrows = get_eyebrow("test_imgs/after5.png") 

    path = './AR_warp_zip/test2/*.bmp'   
    files=glob.glob(path)   
    for file in files:
        print("begin processing..."+file[20:])
        get_eyebrow(file)
    
    print("processing finished!")