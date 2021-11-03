from polygon_check import is_inside_polygon
from ColorConverter import BGR_to_CMYK, BGR_to_HSV
import numpy as np
import mediapipe as mp
import cv2
from sklearn.cluster import DBSCAN, KMeans
import pdb
from matplotlib import pyplot as plt
from UV_utils import uv_coords, uv_map, keypts
import math
import colorsys
import warnings
warnings.simplefilter('error')
import sys
np.set_printoptions(threshold=sys.maxsize)



kernelsize=5
standard_uv_raws=100
standard_uv_cols=100

def get_img_info(img_path):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        image = cv2.imread(img_path)
        # image = cv2.GaussianBlur(image, (7,7), cv2.BORDER_DEFAULT)
        # image = cv2.resize(image, dsize=[480, 480])
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Throw exception if no detected face
        if not results.multi_face_landmarks:
            raise
    landmarks = results.multi_face_landmarks[0].landmark

    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmarks):
        # if ((landmark.HasField('visibility') and
        #     landmark.visibility < VISIBILITY_THRESHOLD) or
        #     (landmark.HasField('presence') and
        #     landmark.presence < PRESENCE_THRESHOLD)):
        #     continue
        x_px = min(math.floor(landmark.x * image_cols), image_cols - 1)
        y_px = min(math.floor(landmark.y * image_rows), image_rows - 1)
        # x_px, y_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
        #                                            image_cols, image_rows)
        idx_to_coordinates[idx] = (x_px, y_px)


    return np.array([[i.x, i.y] for i in landmarks]), idx_to_coordinates, image



"""
traverse mesh_map
"""
def compare_img(img_path_before, img_path_after):
    landmarks1, idx_to_coords1, img1 = get_img_info(img_path_before)
    landmarks2, idx_to_coords2, img2 = get_img_info(img_path_after)
    rows, cols, chanels = img1.shape

    cmyk = BGR_to_CMYK(img1)
    hsv = BGR_to_HSV(img1)
    mk = cmyk[:,:,1]+cmyk[:,:,3]
    mk[mk<0.8]=0
    cv2.imwrite("testcmyk.png", mk*255)
    cv2.imwrite("testhsv.png", hsv[:,:,1]*255)

    silhouette_before = [idx_to_coords1[i] for i in keypts["silhouette"]]
    silhouette_after = [idx_to_coords2[i] for i in keypts["silhouette"]]

    # marker coords on raw image
    markers_before = []
    markers_after = []

    # marker coords on standard UVmap
    uvmarkers = []

    annotated_img1 = img1.copy()
    annotated_img2 = img2.copy()
    
    left, up, right, down = uv_coords[:,0].min(), uv_coords[:,1].min(), uv_coords[:,0].max(), uv_coords[:,1].max()

    # traverse each pixel on mesh map
    print(left, right, up, down)
    for u in np.arange(left, right, (right-left)/70):
        for v in np.arange(up, down, (down-up)/70):

            xy_befores, bgr_befores, hls_befores = [], [], []
            xy_afters, bgr_afters, hls_afters = [], [], []
            uvs = []

            for ustep in np.arange(0, 0.006, 0.003):
                for vstep in np.arange(0, 0.006, 0.003):

                    ratios, indices = get_pt_ref(np.array([u+ustep,v+vstep]), uv_coords)

                    #If part of the area is on the face, align each point to corresponding coords on raw image and standard UVmap
                    if ratios!=[1,0,0]:

                        x_before, y_before = np.matmul(ratios, [idx_to_coords1[indices[0]], 
                                                                idx_to_coords1[indices[1]], 
                                                                idx_to_coords1[indices[2]]])
                        x_before = min(math.floor(x_before), img1.shape[1] - 1)
                        y_before = min(math.floor(y_before), img1.shape[0] - 1)

                        x_after, y_after = np.matmul(ratios, [idx_to_coords2[indices[0]], 
                                                        idx_to_coords2[indices[1]], 
                                                        idx_to_coords2[indices[2]]])
                        x_after = min(math.floor(x_after), img2.shape[1] - 1)
                        y_after = min(math.floor(y_after), img2.shape[0] - 1)

                        # skip if we have processed with this point
                        if (x_before, y_before) in xy_befores or (x_before, y_before) in markers_before or (x_after, y_after) in xy_afters or (x_after, y_after) in markers_after:
                            continue

                        # skip if this point is within silhouette
                        if not is_inside_polygon(points = silhouette_before, p = (x_before, y_before)) or not is_inside_polygon(points = silhouette_after, p = (x_after, y_after)):
                            continue

                        bgr_before = img1[y_before, x_before]
                        hls_before = np.array(colorsys.rgb_to_hls(bgr_before[2]/255, bgr_before[1]/255, bgr_before[0]/255))
                        xy_befores.append((x_before, y_before))
                        bgr_befores.append(bgr_before)
                        hls_befores.append(hls_before)

                        bgr_after = img2[y_after, x_after]
                        hls_after = np.array(colorsys.rgb_to_hls(bgr_after[2]/255, bgr_after[1]/255, bgr_after[0]/255))
                        xy_afters.append((x_after, y_after))
                        bgr_afters.append(bgr_after)
                        hls_afters.append(hls_after)

                        uvs.append((u+ustep, v+vstep))

            # skip if no pt is on face
            if xy_befores:
                avg_bgr_dist = np.sqrt(np.square(np.array(bgr_afters) - np.array(bgr_befores)).sum(axis=1)).sum() / len(bgr_befores)
                hls_dist = abs(np.array(hls_afters) - np.array(hls_befores))
                avg_hdist = hls_dist[:,0].sum()/hls_dist.shape[0]
                avg_ldist = hls_dist[:,1].sum()/hls_dist.shape[0]
                avg_sdist = hls_dist[:,2].sum()/hls_dist.shape[0]
                

                if avg_sdist > 0.17 or avg_ldist > 0.15 or avg_hdist > 5:
                    markers_before.extend(xy_befores)
                    markers_after.extend(xy_afters)
                    uvmarkers.extend(uvs)
                    # cv2.circle(annotated_img1, (x_before, y_before), 0, (255,0,0), 2)
                    # cv2.circle(annotated_img2, (x_after, y_after), 0, (255,0,0), 2)



    #cv2.putText(annotated_img2, str(int(avg_sdiff*10)), (x_after+kernelsize//2, y_after+kernelsize//2), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,255,0), 1, cv2.LINE_AA)
    
    # Do density oriented clustering on UVmap
    clustering = DBSCAN(eps=0.022).fit(uvmarkers)
    # find out which points in uvmarkers form the biggest cluster
    most_freq_idx, biggest_cluster_indices = find_biggest_cluster(clustering.labels_)
    highlights_before = []
    highlights_after = []
    for i in range(len(markers_after)):
        cv2.circle(annotated_img1, markers_before[i], 0, (255,0,0), 2)
        cv2.circle(annotated_img2, markers_after[i], 0, (255,0,0), 2)
    for i in biggest_cluster_indices:
        highlights_before.append(markers_before[i])
        highlights_after.append(markers_after[i])
    highlights_before = np.array(highlights_before)
    highlights_after = np.array(highlights_after)

    cv2.imwrite("markOnBefore_meshtrav.png", annotated_img1)
    cv2.imwrite("markOnAfter_meshtrav.png", annotated_img2)

    # clip highlighted area
    h_left, h_up, h_right, h_down = highlights_after[:,0].min(), highlights_after[:,1].min(), highlights_after[:,0].max(), highlights_after[:,1].max()
    edges = canny_edge(img2[h_up:h_down+1, h_left:h_right+1])
    edges[:,1]+=h_up
    edges[:,0]+=h_left
    for i in edges:
        cv2.circle(img2, i, 0, (0,0,255), 1)
    cv2.imwrite("markOnAfter_mesh_edge.png", img2)



    return highlights_before, highlights_after




def unwrap_face(img, left, right, up, down, landmarks):
    rows, cols, channels = img.shape
    flatFace = np.zeros([standard_uv_raws, standard_uv_cols, 3])
    for x in range(left, right+1):
        for y in range(up, down+1):
            normalized_pt = np.array([x/cols, y/rows])
            ratios, indices = get_pt_ref(normalized_pt, landmarks)
            if ratios != [1,0,0]:
                u, v = np.matmul(ratios, [uv_map[indices[0]], uv_map[indices[1]], uv_map[indices[2]]])
                u = min(math.floor(u), standard_uv_cols - 1)
                v = min(math.floor(v), standard_uv_raws - 1)
                flatFace[v, u] = img[y, x]
    cv2.imwrite("face.png", flatFace)


def preprocessing(img):
    # improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    b = clahe.apply(img[:,:,0])
    g = clahe.apply(img[:,:,1])
    r = clahe.apply(img[:,:,2])
    b = np.expand_dims(b, axis=2)
    g = np.expand_dims(g, axis=2)
    r = np.expand_dims(r, axis=2)
    contrast = np.append(b,g, axis=2)
    contrast = np.append(contrast, r, axis=2)

    
    return contrast


def find_biggest_cluster(clusters):
    """
    :INPUT: clustering.labels_
    :OUTPUT: idx = NO. of biggest cluster, arr = idx of each point in clusters
    :
    """
    dic={}
    for i, num in enumerate(clusters):
        if num == -1:
            continue
        if num in dic.keys():
            dic[num].append(i)
        else:
            dic[num] = [i]
    sorted_dic = sorted(dic.items(), key=lambda x: len(x[1]), reverse=True)
    print([len(i) for i in dic.values()])
    return sorted_dic[0]

def find_top2_clusters(clusters):
    """
    :INPUT: clustering.labels_
    :OUTPUT: tuple of two lists, each contain a cluster
    :
    """
    dic={}
    for i, num in enumerate(clusters):
        if num == -1:
            continue
        if num in dic.keys():
            dic[num].append(i)
        else:
            dic[num] = [i]
    sorted_dic = sorted(dic.items(), key=lambda x: len(x[1]), reverse=True)
    return sorted_dic[0][1], sorted_dic[1][1]



def canny_edge(img):
    # # test thr
    # for i in range(29,40):
    #     for j in range(55,65):
    #         edges=cv2.Canny(img, i, j, L2gradient=True)
    #         print(edges)
    #         cv2.imwrite("test_edge_sat/edgessat_"+str(i)+"_"+str(j)+".png", edges)
    edges = cv2.Canny(img, 29, 60, L2gradient=True).T
    return np.argwhere(edges==255)



def draw_uv(img_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        image = cv2.imread(img_path)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            raise
        annotated_image = image.copy()
        # print(results.multi_face_landmarks[0].landmark[0].x)
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        cv2.imwrite('uvbefore4.png', annotated_image)







def check_lip_lower(landmarks1, landmarks2, idx_to_coords1,
                    idx_tocoords2, img1, img2):
    m = len(keypts["lipsLowerInner"])
    n = len(keypts["lipsLowerOuter"])
    i,j,k = 0,0,1
    idx_to_lip = {"i" : "lipsLowerInner",
                  "j" : "lipsLowerOuter",
                  "k" : "lipsLowerInner"}
    while i < m and i < n and j < m and j < n and k < m and k < n:
        p1 = idx_to_coords1[keypts[idx_to_lip["i"]][i]]
        p2 = idx_to_coords1[keypts[idx_to_lip["j"]][j]]
        p3 = idx_to_coords1[keypts[idx_to_lip["k"]][k]]
        vecA = p2 - p1
        vecB = p3 - p1
        area = np.linalg.norm(np.cross(vecA, vecB))

def get_random_pts_in_triangle(p1, p2, p3, n):
    a = p2 - p1
    b = p3 - p1
    random_uniforms = np.random.rand(n,2)
    random_pts = []
    for u1, u2 in random_uniforms:
        if u1 + u2 > 1:
            u1 = 1 - u1
            u2 = 1 - u2
        w = u1 * a + u2 * b
        random_pts.append(w+p1)
    return random_pts

def get_pt_ref(pt, landmarks_arr, search_ub=10):
    """
    :params pt: np.array([x, y]), the location of the point 
    :params landmarks: landmark obj
    :return pt_ref: [np.array([x1, x2, x3]), np.array[idx1, idx2, idx3]]  //ratio,landmark idx
    :return boolean: True = found the correct solution as pt_ref
    """
    #landmarks_arr = np.array([[i.x, i.y] for i in landmarks])
    dist = np.linalg.norm(landmarks_arr - pt, axis = 1)
    sorted_idx = np.argsort(dist)
    # search_ub = 10
    for i in range(search_ub - 2):
        for j in range(i+1, search_ub - 1):
            for k in range(j+1, search_ub):
                search = [i,j,k] # Indice of triangle's 3 corner
                poly = landmarks_arr[sorted_idx][search]
                if is_in_poly(pt, poly):
                    a = np.array([[poly[0,0], poly[1,0], poly[2,0]],[poly[0,1], poly[1,1], poly[2,1]], [1, 1, 1]])
                    b = np.array([pt[0], pt[1], 1])
                    ans = np.linalg.solve(a,b)
                    pt_ref = [ans.tolist(), sorted_idx[search].tolist()]
                    return pt_ref# Found correct solution
    return [[1, 0, 0], sorted_idx[[0,1,2]].tolist()] # Solution not found. Return the nearest point



def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return: is_in: boolean. True = p in poly
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in

def get_pt_location_from_array_for_polylines(pt_ref, landmarks, w, h):
    """
    :params pt_ref: [np.array([x1, x2, x3]), np.array[idx1, idx2, idx3]]
    :params landmarks: list of double. len = 468. Here the landmarks are UVCoords
    :return np.array([x, y, z])
    """
    landmarks_arr = np.array([[i[0], i[1], 0] for i in landmarks]) #
    ref_ratio, ref_idx = pt_ref
    xy_arr = np.dot(landmarks_arr[ref_idx].T, ref_ratio)
    return [int(xy_arr[0] * w), int(xy_arr[1] * h)]

def draw_keypts(img):
    landmarks1, idx_to_coords1, img1 = get_img_info(img)
    annotated=img1.copy()
    b = 255
    g = 127
    r = 0
    for k,v in keypts.items():
        for i in v:
            cv2.circle(annotated, idx_to_coords1[i], 2, (b,g,r), -1)
        temp=b
        b=g
        g=r
        r=temp
    cv2.imwrite('keypoints.png', annotated)

def try_floodfill(img):
    landmarks1, idx_to_coords1, img1 = get_img_info(img)
    annotated=img1.copy()
    mask=keypts["leftEyeLower0"]

if __name__=="__main__":
    #not using this
    highlights_before, highlights_after = compare_img("test_imgs/before6.png", "test_imgs/after6.png")

    # draw_uv("before4.png")
    
    # landmarks, idx_to_coords, img = get_img_info("after4sat.png")
    # canny_edge(img)

    # img=cv2.imread("after4.png")
    # image_rows, image_cols,c = img.shape
    # coords=[]
    # for i in uv_coords:
    #     x_px = min(math.floor(i[0] * image_cols), image_cols - 1)
    #     y_px = min(math.floor(i[1] * image_rows), image_rows - 1)
    #     # x_px, y_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
    #     #                                            image_cols, image_rows)
    #     coords.append((x_px, y_px))
    #     cv2.circle(img, (x_px, y_px), 0, (255,0,0), 2)
    # cv2.imwrite("uvcoords.png", img)
