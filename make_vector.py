#!/usr/bin/python3
import pickle
import rospy
from sklearn.cluster import DBSCAN
from util import get_rgba_pcd_msg
from sensor_msgs.msg import PointCloud2,Image
import json
import numpy as np
import pandas as pd
import argparse
from pclpy import pcl
import tf
from cv_bridge import CvBridge
import time
import glob
import cv2
import re
from tf import transformations
from scipy.spatial.transform import Rotation as R
from predict import get_colors
import matplotlib.pyplot as plt

# building vector map in osm format
# import argparse
# from datetime import datetime
# import os
# import pathlib
# from bag2way import bag2pose
# from bag2way import pose2line
# from lanelet_xml import LaneletMap

height = {'pole': 5, 'lane':-1.1}

global sempcd
global args
global index
global poses
global br
global savepcd
global odom_trans
global last_points
global vectors
global lanepcd

class myqueue(list):
    def __init__(self, cnt=-1):
        self.cnt = cnt
        self.index = 0

    def append(self, obj):
        self.index+=1
        if len(self) >= self.cnt and self.cnt != -1:
            self.remove(self[0])
        super().append(obj)

    def is_empty(self):
        if len(self) == 0:
            return True
        else:
            return False

def color2int32(tup):
    return np.array([*tup[1:], 255]).astype(np.uint8).view('uint32')[0]


def class2color(cls,alpha = False):
    c = color_classes[cls]
    if not alpha:
        return np.array(c).astype(np.uint8)
    else:
        return np.array([*c, 255]).astype(np.uint8)

def save_nppc(nparr,fname):
    s = nparr.shape
    if s[1] == 4:#rgb
        tmp = pcl.PointCloud.PointXYZRGBA(nparr[:,:3],np.array([color_classes[int(i)] for i in nparr[:,3]]))
    else:
        tmp = pcl.PointCloud.PointXYZ(nparr)
    pcl.io.save(fname,tmp)
    return tmp

def draw_line(p1,p2):
    assert isinstance(p1,np.ndarray) or isinstance(p1,set)
    assert isinstance(p2,np.ndarray) or isinstance(p2,set)
    assert p1.shape == p2.shape
    if len(p1.shape) == 2 or p1.shape[0]== 2:
        d = np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2)
        n = int(d/0.01)
        x = np.linspace(p1[0],p2[0],n)
        y = np.linspace(p1[1],p2[1],n)
        line = np.stack((x,y),axis=1)
    else:
        d = np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2+(p1[2]-p2[2])**2)
        n = int(d/0.01)
        x = np.linspace(p1[0],p2[0],n)
        y = np.linspace(p1[1],p2[1],n)
        z = np.linspace(p1[2],p2[2],n)
        line = np.stack((x,y,z),axis=1)
    return line


def pcd_trans(pcd,dt,dr,inverse = False):
    length = len(pcd)
    if not isinstance(pcd,np.ndarray):
        pcd = np.array(pcd)
    pcd = pcd.T
    pcd_xyz = pcd[:3]
    ones = np.ones((1, length))
    transpcd = np.vstack((pcd_xyz, ones))
    # print(dt)
    # print(transformations.translation_matrix(dt))
    mat44 = np.dot(transformations.translation_matrix(dt), transformations.quaternion_matrix(dr))
    if inverse:
        mat44 = np.matrix(mat44).I
    pcd[:3] = np.dot(mat44, transpcd)[:3]
    transedpcd = pcd.T
    return transedpcd

def point_trans(ptpose,ori_frame,inverse = False):
    rotp = pd.Series(ptpose[3:7], index=['x', 'y', 'z', 'w'])
    dt = ori_frame
    dr = pd.Series(ori_frame[3:7], index=['x', 'y', 'z', 'w'])
    mat44 = np.dot(transformations.translation_matrix(dt), transformations.quaternion_matrix(dr))
    Fp = np.dot(transformations.translation_matrix(ptpose), transformations.quaternion_matrix(rotp))
    if inverse:
        mat44 = np.matrix(mat44).I
    transpose = np.dot(mat44, Fp)
    # print("transpose",transpose)
    pt_s = np.asarray([transpose[0,3],transpose[1,3],transpose[2,3]])
    r = R.from_matrix(transpose[:3,:3])
    pt_s = np.append(pt_s, r.as_quat())
    # pt_s = np.squeeze(pt_s)
    return pt_s


def get_lane_centers(pcd):
    pcd = pcd[(pcd[:,0]>-10)&(pcd[:,0]<10)]
    centers = []
    if len(pcd) == 0:
        return centers
    dbs.fit(pcd)
    labels = dbs.fit_predict(pcd)  # label   
    cluster = list(set(labels))
    # print(np.shape(pcd))
    n = len(cluster)
    # print("num of clustering:", cluster)
    for i in cluster:
        if n <= 0:
            continue
        c = pcd[labels == i]  # each cluster
        if abs(c[:,0].max()-c[:,0].min()) > 0.3:
            if c[:,0].mean() < 0:
                center = np.array((c[:, 0].max()-0.2, c[:, 1].mean(), c[:,2].mean()))#-2.3))这里假定都是直道
            else:
                center = np.array((c[:, 0].min()+0.2, c[:, 1].mean(), c[:,2].mean()))#-2.3))
        else:
            center = np.array((c[:,0].mean(),c[:,1].mean(),-2.3))
        centers.append(center)
    return centers

def get_vector_nodes(pcd):
    global pre_left_node_dis
    global pre_right_node_dis
    global default_half_road_width
    global last_pose

    if((last_pose[0] == 0) & (last_pose[1] == 0) & (last_pose[2] == 0)):
        last_pose = poses[index][:3]

    # initialize
    alpha = 0.85
    # alpha is parameter to combine semantic information and pose information
    leftnode = []
    rightnode = []
    pcd = pcd[(pcd[:,2]<20)&(pcd[:,0]>-10)&(pcd[:,0]<10)]
    if (len(pcd) == 0):
        return leftnode, rightnode
    if (np.linalg.norm(poses[len(poses)-1][:3] - poses[index][:3]) < 5.5):
        return leftnode, rightnode
    if (np.linalg.norm(last_pose - poses[index][:3]) < 0.5):
        return leftnode, rightnode
    
    last_pose = poses[index][:3]
    p = poses[index]   
    this_index = index + 1
    
    while True:
        dis_2pose = np.linalg.norm(poses[this_index][:3] - poses[index][:3])
        if dis_2pose > 4.5:
            break   
        this_index += 1      
    # calculate pose after 2 seconds in this frame
    pose_extend = poses[this_index]
    # print("index info", index, this_index)
    rot0 = pd.Series(p[3:7], index=['x', 'y', 'z', 'w'])
    pose_estimate = point_trans(pose_extend, p, True) # 反变换到Lidar坐标系下
    pose_estimate[2] = pose_estimate[2] - 0.9
    
    left_tran = [0, pre_left_node_dis, 0, 0, 0, 0, 1]
    right_tran = [0, -pre_right_node_dis, 0, 0, 0, 0, 1]
    leftnode_estimate = point_trans(left_tran,pose_estimate)
    rightnode_estimate = point_trans(right_tran,pose_estimate)
    # return leftnode_estimate, rightnode_estimate
    # print("pose_estimation",pose_estimate)
    # print("leftnode_estimation",leftnode_estimate)
    # print("rightnode_estimation",rightnode_estimate)

    # DBSCAN聚类
    dbs.fit(pcd)
    labels = dbs.fit_predict(pcd)  # label   
    cluster = list(set(labels))
    n = len(cluster)
    print("num of clustering:", cluster)

    # ransac特征提取
    # valid_cluster = []
    # for i in cluster:
    #     if n <= 0:
    #         continue
    #     c = pcd[labels == i]  # each cluster

    # 构建左右节点  
    # # 预测节点到聚类距离 
    pt2pcd_dis = []
    # plt.scatter(pcd[:,0],pcd[:,1], s = 2, label='Lane points', c = (0,0,0.5))
    for i in cluster:
        if len(cluster) <= 0:
            continue
        c = pcd[labels == i]
        pc = c[:,:3]
        # print(np.shape(pc))
        if(len(pc)>50&i>-1):
            pdis,pp,pclose = dis3d_pt2pcd(pose_estimate,pc)
            ldis,lp,lclose = dis3d_pt2pcd(leftnode_estimate,pc)
            rdis,rp,rclose = dis3d_pt2pcd(rightnode_estimate,pc)
            # plt.scatter(pp[0],pp[1],label='1', c = (1,0,0))
            # plt.scatter(lp[0],lp[1],label='2', c = (0.8,0,0))
            # plt.scatter(rp[0],rp[1],label='3', c = (0.6,0,0))
            # plt.scatter(pc[:,0],pc[:,1], s = 1, c = (0,0,(0.7+0.6*i-int(0.7+0.6*i))))
            pt2pcd_dis.append([pdis,ldis,rdis])
     
    # plt.scatter(pose_estimate[0],pose_estimate[1],label='Reference point',c = (0,1,0))
    # plt.scatter(leftnode_estimate[0],leftnode_estimate[1],label='Priori info estimation',c = (0,0.5,0))
    # plt.scatter(rightnode_estimate[0],rightnode_estimate[1],c = (0,0.5,0))
    
    # plt.legend()
    # plt.show()
    if len(pt2pcd_dis) <=0:
        return leftnode, rightnode
    
    pt2pcd_dis = np.asarray(pt2pcd_dis)    
    
    args_min = np.argmin(pt2pcd_dis, axis=0)

    # print(args_min)

    sem_lerror = pt2pcd_dis[args_min[1],1]
    sem_ldis = pt2pcd_dis[args_min[1],0]
    sem_rerror = pt2pcd_dis[args_min[2],2]
    sem_rdis = pt2pcd_dis[args_min[2],0]

    print(sem_ldis,sem_lerror,sem_rdis,sem_rerror)
    # pose correction constant
    pL = pre_left_node_dis - default_half_road_width
    pR = pre_right_node_dis - default_half_road_width

    
   
    if(-0.5 < 2 * default_half_road_width - sem_ldis - sem_rdis < 0.5):
        print('good shape')
    else:
        if ((-1 < default_half_road_width - sem_ldis < 0.5) | (-1 < default_half_road_width - sem_rdis < 0.5)):
            print ('one in good shape')

            if ((-1 < default_half_road_width - sem_ldis < 0.5) & (-1 < default_half_road_width - sem_rdis < 0.5)):
                if(min(sem_lerror, sem_rerror) < 0.3):
                    if(sem_lerror < sem_rerror):
                        sem_rdis = (pre_left_node_dis + pre_right_node_dis - sem_ldis)
                    else:
                        sem_ldis = (pre_left_node_dis + pre_right_node_dis - sem_rdis)
                else:
                    sem_ldis = pre_left_node_dis
                    sem_rdis = pre_right_node_dis

            else:     
                if(-1 < default_half_road_width - sem_ldis < 0.5):
                    if(sem_lerror < 0.3):
                        sem_rdis = (pre_left_node_dis + pre_right_node_dis - sem_ldis)
                    else:
                        sem_ldis = pre_left_node_dis
                        sem_rdis = pre_right_node_dis

                else:
                    if(sem_rerror < 0.3):
                        sem_ldis = (pre_left_node_dis + pre_right_node_dis - sem_rdis)
                    else:
                        sem_ldis = pre_left_node_dis
                        sem_rdis = pre_right_node_dis  
        else:
            print('sem fail')
            sem_ldis = pre_left_node_dis
            sem_rdis = pre_right_node_dis 

    # four multiple contraints

    ldis = alpha * pre_left_node_dis + (1-alpha) * sem_ldis - (0.05+0.25*np.abs(pL+pR)) * pL 
    rdis = alpha * pre_right_node_dis + (1-alpha) * sem_rdis - (0.05+0.25*np.abs(pL+pR)) * pR 
    # ldis = sem_ldis
    # rdis = sem_rdis
    
    left_tran = [0, ldis, 0, 0, 0, 0, 1]
    right_tran = [0, -rdis, 0, 0, 0, 0, 1]
    leftnode = point_trans(left_tran,pose_estimate)
    rightnode = point_trans(right_tran,pose_estimate)
    pre_left_node_dis = ldis
    pre_right_node_dis = rdis
    # plt.scatter(leftnode[0],leftnode[1],label='Predicted node',c = (1,0,0))
    # plt.scatter(rightnode[0],rightnode[1],c = (1,0,0))
    # # print("right distance",rdis, "left distance", ldis)
    # plt.legend()
    # plt.show()

    return leftnode, rightnode

def dis3d_pt2pcd(single_pt,pcd,down_rate = 10):
    pt_total_dis = np.linalg.norm(pcd - single_pt[:3], axis=1)
    len_min = int(len(pcd)/down_rate)
    arg_min_pt = np.argsort(pt_total_dis)

    min_dis_pts = []
    for i in range(len_min):
        min_dis_pts.append(pcd[arg_min_pt[i]])
    min_dis_pts = np.asarray(min_dis_pts)    

    mean_pt = np.average(min_dis_pts, axis = 0)
    mean_pt_dis = np.linalg.norm(mean_pt - single_pt[:3])

    # return mean_pt_dis, min_dis_pts, mean_pt
    return mean_pt_dis, mean_pt, pt_total_dis[arg_min_pt[0]]

def get_pole_centers(pcd):
    centers = []
    if len(pcd) == 0:
        return centers
    pole_dbs.fit(pcd)
    labels = pole_dbs.fit_predict(pcd[:,:2])  # label
    cluster = list(set(labels))
    n = len(cluster)
    for i in cluster:
        if n <= 0:
            continue
        c = pcd[labels == i]  # each cluster
        center = np.array((c[:,0].mean(),c[:,1].mean(),c[:,2].max()))
        centers.append(center)
    return centers


def process():
    global sempcd
    global args
    global index
    global poses
    global br
    global last_points
    global vectors
    global lanepcd
    global linepoint
    global left_nodes
    global right_nodes
    global pre_left_node_dis
    global pre_right_node_dis

    if args.trajectory:
        p = poses[index]
        print("This is the loop: ",index)
        rotation = pd.Series(p[3:7], index=['x', 'y', 'z', 'w'])
        br.sendTransform((p[0], p[1], p[2]), rotation, rospy.Time(time.time()), 'odom', 'world')
        if args.vector:
            # pole can be vectorized globally
            # process lane
            lanes = sempcd[sempcd[:, 3] == config['lane_class']]
            if len(lanepcd) < window:
                lanepcd.append(lanes)
            else:
                lanepcd.append(lanes)
                if index % step == 0:
                    lanes = np.vstack(lanepcd)
                    lanes = pcd_trans(lanes, p, rotation, True)
                    lanes = lanes[lanes[:, 1] < 8] #3 for parking lot, 8 for science park
                    #testPubHandle.publish(get_rgba_pcd_msg(pcd_trans(lanes,p,rotation)))
                    # centers = get_lane_centers(lanes)
                    centers = []
                    # print(centers)
                    lnode,rnode = get_vector_nodes(lanes)
                    # print("left node", lnode)
                    # print("right node", rnode)
                    
                    if len(lnode) != 0:
                        lnode_world = point_trans(lnode,p)
                        rnode_world = point_trans(rnode,p)
                        # print("current positon", p)
                        # print("left node", lnode_world)
                        # print("right node", rnode_world)
                        left_nodes=np.append(left_nodes,[lnode_world[:3]],axis=0)
                        right_nodes=np.append(right_nodes,[rnode_world[:3]],axis=0)
                        # print(np.shape(left_nodes),np.shape(right_nodes))

                    if len(centers) != 0:                       
                        centers = list(pcd_trans(centers,p,rotation))
                        # print(centers)
                        # print(p[0:6])
                        if last_points:
                            pairs = {}
                            lines = []
                            for i in sum(last_points,[]):
                                d = 100000
                                pair = (d, None, None)
                                for j in centers:
                                    dis = np.linalg.norm(i - j)
                                    if dis > 1:
                                        continue
                                    if d != min(d, dis):
                                        d = dis
                                        pair = (d, i, j)
                                if pair[2] is None:
                                    continue
                                if tuple(pair[2]) in pairs:
                                    if d < pairs[tuple(pair[2])][0]:
                                        pairs[tuple(pair[2])] = pair
                                else:
                                    pairs[tuple(pair[2])] = pair
                            pairs_all.append(pairs)
                            # print(pairs)
                            
                            tmplinedata = np.empty((0,3))
                            for i in pairs:
                                lines.append(draw_line(*(pairs[i][1:])))
                                tmplinedata = np.append(tmplinedata,pairs[i][1:2],axis=0)
                            # print(tmplinedata)
                            print(np.shape(tmplinedata))   
                            if(np.shape(tmplinedata)[0]<3):
                                print("The line didn't in number 3\n\n")                   
                            linepoint=np.append(linepoint, tmplinedata,axis=0)   
                            # print(np.shape(linepoint))
                            if len(lines) != 0:
                                lines = np.vstack(lines)
                                #lines = pcd_trans(lines,p,rotation)
                                vectors.append(lines)
                                vecmsg = get_rgba_pcd_msg(lines)
                                vecmsg.header.frame_id = 'world'
                                vecPubHandle.publish(vecmsg)
                        last_points.append(centers)
                   
    index += 1


    if args.filters:
        sempcd = sempcd[np.in1d(sempcd[:, 3], args.filters)]
    sem_msg = get_rgba_pcd_msg(sempcd)
    sem_msg.header.frame_id = 'world'
    semanticCloudPubHandle.publish(sem_msg)

    if args.semantic and index < len(simgs):
        simg = cv2.imread(simgs[index],0)
        semimg = colors[simg.flatten()].reshape((*simg.shape,3))
        semimgPubHandle.publish(bri.cv2_to_imgmsg(semimg, 'bgr8'))
    if args.origin and index < len(imgs):
        imgPubHandle.publish(bri.cv2_to_imgmsg(cv2.imread(imgs[index]), 'bgr8'))


# parse arguments
parser = argparse.ArgumentParser(description='Rebuild semantic point cloud')
parser.add_argument('-c','--config',help='The config file path, recommand use this method to start the tool')
parser.add_argument('-i','--input',type=argparse.FileType('rb'))
parser.add_argument('-m','--mode',choices=['outdoor','indoor'],help="Depend on the way to store the pickle file")
parser.add_argument('-f','--filters', default=None,nargs='+',type=int,help='Default to show all the classes, the meaning of each class refers to class.json')
parser.add_argument('-s','--save',default=None,help='Save to pcd file')
parser.add_argument('-t','--trajectory',default=None,help='Trajectory file, use to follow the camera')
parser.add_argument('--semantic',default=None,help='Semantic photos folder')
parser.add_argument('--origin',default=None,help='Origin photos folder')
parser.add_argument('--vector',default=None,help='Do the vectorization, only available when filters are accepted',action='store_true')
args = parser.parse_args()

if args.config:
    with open(args.config,'r') as f:
        config = json.load(f)

args.input = (args.input or open(config['save_folder']+(config['mode'] == 'indoor' and '/indoor.pkl' or '/outdoor.pkl'),'rb'))
args.mode = (args.mode or config['mode'])
args.trajectory = (args.trajectory or config['save_folder']+'/pose.csv')
args.save = (args.save or config['save_folder']+'/result.pcd')
args.semantic = (args.semantic or config['save_folder']+'/sempics')
args.origin = (args.origin or config['save_folder']+'/originpics')
args.vector = (args.vector) or config['vector']
# init variables

window = 10
step = 1
default_half_road_width = 1.5
pre_left_node_dis = default_half_road_width
pre_right_node_dis = default_half_road_width
last_pose = np.asarray([0,0,0])


# start ros
rospy.init_node('fix_distortion', anonymous=False, log_level=rospy.DEBUG)
semanticCloudPubHandle = rospy.Publisher('SemanticCloud', PointCloud2, queue_size=5)
vecPubHandle = rospy.Publisher('VectorCloud', PointCloud2, queue_size=5)
testPubHandle = rospy.Publisher('TestCloud', PointCloud2, queue_size=5)
semimgPubHandle = rospy.Publisher('SemanticImg',Image,queue_size = 5)
imgPubHandle = rospy.Publisher('Img',Image,queue_size = 5)

color_classes = get_colors(config['cmap'])
savepcd = []
vectors = []
linepoint = np.empty((0,3))
left_nodes = np.empty((0,3))
right_nodes = np.empty((0,3))
bri = CvBridge()
index = 0
br = tf.TransformBroadcaster()
dbs = DBSCAN(eps = 0.5,min_samples=80,n_jobs=24)
pole_dbs = DBSCAN(eps = 0.5,min_samples=50,n_jobs=24)
#dbs = DBSCAN()
last_points = myqueue(1)
lanepcd = myqueue(window)
polepcd = myqueue(window)
pairs_all = []
vec_world = []


if args.semantic:
    simgs = glob.glob(args.semantic+'/*')
    simgs.sort()
    #simgs.sort(key = lambda x:int(re.findall('[0-9]{4,5}',x)[0]))
    colors = color_classes.astype('uint8')

if args.origin:
    imgs = glob.glob(args.origin+'/*')
    imgs.sort()
    #imgs.sort(key = lambda x:int(re.findall('[0-9]{4,5}',x)[0]))

if args.trajectory:
    poses = np.loadtxt(args.trajectory, delimiter=',')


if args.mode == 'indoor':
    sempcds = pickle.load(args.input)
    for sempcd in sempcds:
        process() 
    # np.save("/home/gjd/hdmap_ws/src/HDMap/linepoint.npy",linepoint)
    np.save("/home/gjd/hdmap_ws/src/HDMap/left_nodes.npy",left_nodes)
    np.save("/home/gjd/hdmap_ws/src/HDMap/right_nodes.npy",right_nodes)
    np.save("/home/gjd/hdmap_ws/src/HDMap/linepoint.npy",linepoint)
    plt.rcParams.update({'font.size': 24})
    plt.scatter(left_nodes[:,0],left_nodes[:,1], label = 'Left lane nodes',s = 1, color = 'b')
    plt.scatter(right_nodes[:,0],right_nodes[:,1], label = 'Right lane nodes',s = 1, color = 'g')
    plt.scatter(poses[:,0],poses[:,1], s = 1, label = 'Pose trajectory', color = 'r')
    plt.legend()
    plt.show()
    savepcd = np.concatenate(sempcds)
    print('done')
   
elif args.mode == 'outdoor':
    try:
        while True:
            sempcd = pickle.load(args.input)
            savepcd.append(sempcd)
            process()
            print("this is the loop:",index)
    except EOFError:
        print('done')
        np.save("/home/gjd/hdmap_ws/src/HDMap/linepoint.npy",linepoint)
        savepcd = np.concatenate(savepcd)

if args.vector:
    poles = savepcd[np.in1d(savepcd[:, 3], [config['pole_class']])]
    poles = poles[poles[:,2]>-2]
    poles = poles[poles[:,2]<8]
    pole_centers = get_pole_centers(poles[:, :3])
    poles = []
    for pole in pole_centers:
        poles.append(draw_line(pole,np.array([pole[0],pole[1],-2.3])))
    polemsg = get_rgba_pcd_msg(np.vstack(poles),color2int32((255,0,255,0)))
    vecPubHandle.publish(polemsg)


if args.save is not None:
    save_nppc(savepcd,args.save)
    lane = np.vstack(vectors)
    p = np.vstack(poles)
    v = np.vstack((lane,p))
    save_nppc(v,'/'.join(args.save.split('/')[:-1])+'/vector.pcd')



def filter():
    srcpcd = './worldSemanticCloud.pcd'
    wsc = pcl.PointCloud.PointXYZRGBA()
    fpc = pcl.PointCloud.PointXYZRGBA()
    pcl.io.loadPCDFile(srcpcd,wsc)
    f = pcl.filters.RadiusOutlierRemoval.PointXYZRGBA()
    f.setInputCloud(wsc)
    f.setRadiusSearch(0.1)
    f.setMinNeighborsInRadius(10)
    f.filter(fpc)
