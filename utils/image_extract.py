import os.path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import warnings
warnings.simplefilter('ignore')

def get_default_angles():
    angles = [(0, 90), (0, -90)]
    for i in range(1, 4):
        angles += [(0 + 45 * j, -90 + 45 * i) for j in range(8)]
    return angles

def imresize(im, sz):
    if np.amax(im) <= 1.0:
        im = im * 255
        scl = 255.0
    else:
        scl = 1
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize((sz[1], sz[0]))) / scl

def rnd(x):
    if type(x) is np.ndarray:
        return (x + 0.5).astype(np.int)
    else:
        return round(x)

def polar(cord):
    if cord.ndim == 1:
        P = np.linalg.norm(cord)
    else:
        P = np.linalg.norm(cord, axis=0)
    phi = np.arcsin(cord[2] / P)
    theta_positive = np.arccos(cord[0] / np.sqrt(cord[0]**2 + cord[1]**2))
    theta_negative = - np.arccos(cord[0] / np.sqrt(cord[0]**2 + cord[1]**2))
    theta = (cord[1] > 0) * theta_negative + (cord[1] <= 0) * theta_positive
    return [theta, phi]


class CameraPrm:
    # camera_angle, view_angle: [horizontal, vertical]
    # L: distance from camera to image plane
    def __init__(self, camera_angle, image_plane_size=None, view_angle=None, L=None):
        # camera direction (in radians) [horizontal, vertical]
        self.camera_angle = camera_angle

        if camera_angle is not None:
            camera_angle = tuple([np.deg2rad(i) for i in camera_angle])
        if view_angle is not None:
            view_angle = tuple([np.deg2rad(i) for i in view_angle])

        # view_angle: angle of view in radians [horizontal, vertical]
        # image_plane_size: [image width, image height]
        if view_angle is None:
            self.image_plane_size = image_plane_size
            self.L = L
            self.view_angle = 2.0 * np.arctan(np.array(image_plane_size) / (2.0 * L))
        elif image_plane_size is None:
            self.view_angle = view_angle
            self.L = L
            self.image_plane_size = 2.0 * L * np.tan(np.array(view_angle) / 2.0)
        else:
            self.image_plane_size = image_plane_size
            self.view_angle = view_angle
            L = (np.array(image_plane_size) / 2.0) / np.tan(np.array(view_angle) / 2.0)
            if rnd(L[0]) != rnd(L[1]):
                print('Warning: image_plane_size and view_angle are not matched.')
                va = 2.0 * np.arctan(np.array(image_plane_size) / (2.0 * L[0]))
                ips = 2.0 * L[0] * np.tan(np.array(view_angle) / 2.0)
                print('image_plane_size should be (' + str(ips[0]) + ', ' + str(ips[1]) +
                      '), or view_angle should be (' + str(math.degrees(va[0])) + ', ' + str(
                    math.degrees(va[1])) + ').')
                # return
            self.L = L[0]

        # unit vector of cameara direction
        self.nc = np.array([
            np.cos(camera_angle[1]) * np.cos(camera_angle[0]),
            -np.cos(camera_angle[1]) * np.sin(camera_angle[0]),
            np.sin(camera_angle[1])
        ])

        # center of image plane
        self.c0 = self.L * self.nc

        # unit vector (xn, yn) in image plane
        self.xn = np.array([
            -np.sin(camera_angle[0]),
            -np.cos(camera_angle[0]),
            0
        ])
        self.yn = np.array([
            -np.sin(camera_angle[1]) * np.cos(camera_angle[0]),
            np.sin(camera_angle[1]) * np.sin(camera_angle[0]),
            np.cos(camera_angle[1])
        ])

        # meshgrid in image plane
        [c1, r1] = np.meshgrid(np.arange(0, rnd(self.image_plane_size[0])), np.arange(0, rnd(self.image_plane_size[1])))

        # 2d-cordinates in image [xp, yp]
        img_cord = [c1 - self.image_plane_size[0] / 2.0, -r1 + self.image_plane_size[1] / 2.0]

        # 3d-cordinatess in image plane [px, py, pz]
        self.p = self.get_3Dcordinate(img_cord)

        # polar cordinates in image plane [theta, phi]
        self.polar_omni_cord = polar(self.p)

    def get_3Dcordinate(self, c):
        [xp, yp] = c
        if type(xp) is np.ndarray:  # xp, yp: array
            return xp * self.xn.reshape((3, 1, 1)) + yp * self.yn.reshape((3, 1, 1)) + np.ones(
                xp.shape) * self.c0.reshape((3, 1, 1))
        else:  # xp, yp: scalars
            return xp * self.xn + yp * self.yn + self.c0

def get_index_params(camera_prm,omni_size,imp_shape):
    [c_omni, r_omni] = np.meshgrid(np.arange(omni_size[0]), np.arange(omni_size[1]))
    theta = (2.0 * c_omni / float(omni_size[0] - 1) - (omni_size[0]-1)/omni_size[0]) * np.pi
    phi = (0.5 - r_omni / float(omni_size[1] - 1)) * np.pi

    pn = np.array([
        np.cos(phi) * np.cos(theta),
        -np.cos(phi) * np.sin(theta),
        np.sin(phi)
    ])
    pn = pn.transpose(1, 2, 0)

    # camera parameters
    L = camera_prm.L
    nc = camera_prm.nc
    xn = camera_prm.xn
    yn = camera_prm.yn
    w1 = camera_prm.image_plane_size[0]
    h1 = camera_prm.image_plane_size[1]

    # True: inside image (candidates), False: outside image
    cos_alpha = np.dot(pn, nc)
    mask = cos_alpha >= 2 * L / np.sqrt(w1 ** 2 + h1 ** 2 + 4 * L ** 2)  # circle

    r = np.zeros((omni_size[1], omni_size[0]))
    xp = np.zeros((omni_size[1], omni_size[0]))
    yp = np.zeros((omni_size[1], omni_size[0]))
    r[mask == True] = L / np.dot(pn[mask == True], nc)
    xp[mask == True] = r[mask == True] * np.dot(pn[mask == True], xn)
    yp[mask == True] = r[mask == True] * np.dot(pn[mask == True], yn)

    # True: inside image, False: outside image
    mask = (mask == True) & (xp > -w1 / 2.0) & (xp < w1 / 2.0) & (yp > -h1 / 2.0) & (yp < h1 / 2.0)
    xp[mask == False] = 0
    yp[mask == False] = 0
    rate0 = (imp_shape[1]-1)/imp_shape[1]
    rate1 = (imp_shape[2]-1)/imp_shape[2]
    [r1, c1] = np.array([xp/imp_shape[2]*2*rate1,-yp/imp_shape[1]*2*rate0])
    return mask,np.stack([r1,c1],axis=-1)

class CoordinateEmbedding:
    def __init__(self, omni_size,device="cpu",index_counts=26):
        self.omni_size = [i+2 for i in omni_size]
        self.coordinate = torch.zeros((2,self.omni_size[1],self.omni_size[0]),device=device)
        self.mask = torch.zeros((self.omni_size[1],self.omni_size[0],),device=device)
        self.index_counts = index_counts

    def add_coordinate(self, camera_prm,inp_shape,index=0):
        mask, pos = get_index_params(camera_prm, self.omni_size, inp_shape)
        mask = torch.from_numpy(mask).to(bool).detach()
        pos = torch.from_numpy(pos).to(torch.float32).permute(2,0,1)
        pos[1] = pos[1]/self.index_counts+(2*index-self.index_counts+1)/self.index_counts
        self.coordinate[:,mask] = pos[:,mask]
        self.mask = torch.clip(self.mask + mask,0,1)

    def get_coordinage(self):
        return self.coordinate.permute(1,2,0).unsqueeze(0)[:,1:-1,1:-1],self.mask.unsqueeze(0).unsqueeze(0)[:,:,1:-1,1:-1]

def convert_extract_image_coordinates(angle=get_default_angles(),target_dicts=[0],
                                      image_shape=256,view_angle=60,erp_image_shape=(1024,512)):
    coors = CoordinateEmbedding(erp_image_shape,index_counts=len(angle))
    for t in target_dicts:
        phi, theta = angle[t]
        coordinate = CameraPrm(camera_angle=(phi, theta,), view_angle=(view_angle, view_angle),
                               image_plane_size=(image_shape, image_shape))
        coors.add_coordinate(coordinate,(1,image_shape,image_shape),t)
    return coors.get_coordinage()

class _ExtractImageConvertor:
    def __init__(self,angle=get_default_angles(),target_dicts=[0],
                 image_shape=256,view_angle=45,erp_image_shape=(1024,512)):
        super().__init__()
        self.angle = angle
        self.coors,self.masks = convert_extract_image_coordinates(angle,target_dicts,image_shape,view_angle,erp_image_shape)

    def __call__(self, img):
        if img.device != self.coors.device:
            self.coors = self.coors.to(img.device)
            self.masks = self.masks.to(img.device)
        img = rearrange(img,"(b t) c h w->b c (t h) w",t=len(self.angle))
        img = F.grid_sample(img, self.coors.repeat(img.shape[0],1,1,1), mode="bilinear", align_corners=True)
        return img*self.masks,self.masks

class ExtractImageConvertor:
    def __init__(self,image_shape=256,view_angle=45,erp_image_shape=(1024,512),
                 dist_func_name="l2",cost_func_name="linear"):
        super().__init__()
        if dist_func_name=="l1":
            _dist_func = lambda x1,x2:(torch.abs(x1) + torch.abs(x2)) / 2
        elif dist_func_name=="sup":
            _dist_func = lambda x1,x2:torch.maximum(torch.abs(x1),torch.abs(x2))
        else:
            _dist_func = lambda x1,x2:np.sqrt((x1 ** 2 + x2 ** 2) / 2)
        dist_func = lambda x1,x2:torch.clip(1 - _dist_func(x1,x2),1e-5,1)

        if cost_func_name=="exp":
            _cost_func = lambda x:2**x-1
        elif cost_func_name=="square":
            _cost_func = lambda x:x**2
        elif cost_func_name=="cubic":
            _cost_func = lambda x:x**3
        elif cost_func_name=="cosine":
            _cost_func = lambda x:1-torch.cos(0.5*x*torch.pi)
        elif cost_func_name=="square_root":
            _cost_func = lambda x:torch.sqrt(x)
        elif cost_func_name=="logarithmic":
            _cost_func = lambda x:torch.log(x+1)/np.log(2)
        else:
            _cost_func = lambda x:x
        cost_func = lambda x:torch.clip(_cost_func(x),1e-5,1)


        eic_datas = [
            [0, 1, 10, 12, 14, 16],
            [11, 13, 15, 17],
            [2, 6, 18, 22],
            [3, 7, 19, 23],
            [4, 8, 20, 24],
            [5, 9, 21, 25],
        ]
        self.eics = [_ExtractImageConvertor(angle=get_default_angles(),target_dicts=i,
                                           image_shape=image_shape,view_angle=view_angle,erp_image_shape=erp_image_shape)
                     for i in eic_datas]
        corr = torch.meshgrid(torch.linspace(-1,1,256),torch.linspace(-1,1,256))
        cost = cost_func(dist_func(corr[0],corr[1]))
        cost_data_img = cost.unsqueeze(0).unsqueeze(0).repeat(26,3,1,1)
        cost_erp = [eic(cost_data_img) for eic in self.eics]
        cost_sums = 0
        for i in cost_erp:
            cost_sums += i[0]*i[1]
        self.cost = [i[0]*i[1]/cost_sums for i in cost_erp]

    def __call__(self, img):
        img_mask_data = [eic(img)[0] for eic in self.eics]
        if self.cost[0].device != img_mask_data[0].device:
            self.cost = [i.to(img_mask_data[0].device) for i in self.cost]
        merge_image = sum([img_mask_data[i]*self.cost[i%26] for i in range(len(img_mask_data))])
        return torch.clip(merge_image,0,1)



class ImageExtraction():
    def __init__(self,angle=get_default_angles(),image_shape=256,view_angle=45,erp_image_shape=(1024,512)):
        super().__init__()
        self.angle = angle
        self.image_shape = image_shape
        self.view_angle = view_angle
        self.erp_image_shape = erp_image_shape
        self.calc_erp2extract_coordinates()

    def calc_erp2extract_coordinates(self):
        ls = []
        for phi, theta in self.angle:
            coordinate = CameraPrm(camera_angle=(phi, theta,), view_angle=(self.view_angle, self.view_angle),
                                   image_plane_size=(self.image_shape, self.image_shape)).polar_omni_cord
            coordinate[0] = coordinate[0] / np.pi
            coordinate[1] = -2 * coordinate[1] / np.pi
            ls.append(np.array(coordinate))
        coordinate = np.concatenate(ls,axis=1)
        self.coordinate = torch.from_numpy(coordinate).unsqueeze(0).permute(0, 2, 3, 1)
        self.coordinate.requires_grad = False
        self.coordinate[:,:,:,0] = self.coordinate[:,:,:,0] * ((self.erp_image_shape[0]-1)/self.erp_image_shape[0])
        self.coordinate[:,:,:,1] = self.coordinate[:,:,:,1] * ((self.erp_image_shape[1]-1)/self.erp_image_shape[1])
        self.coordinate = self.coordinate.to(torch.float32)

    def __call__(self,inputs,reshape=True):
        if len(inputs.shape)==3:
            inputs = inputs.unsqueeze(0)
        if inputs.device != self.coordinate.device:
            self.coordinate = self.coordinate.to(inputs.device)
        inputs = F.pad(inputs,(0,0,1,1),mode="circular")
        inputs = F.pad(inputs,(1,1,0,0),mode="reflect")
        extract = F.grid_sample(inputs, self.coordinate.repeat(inputs.shape[0],1,1,1),mode="bilinear",align_corners=True)
        if reshape:
            extract = rearrange(extract,"b c (t h) w->(b t) c h w",t=len(self.angle))
        return extract


class ImageEmbedding():
    def __init__(self,phi,theta,angles=get_default_angles(),image_shape=256,view_angle=45,
                 extract_image_shape=256,extract_view_angle=45,erp_image_shape=(1024,512)):
        super().__init__()
        coors = CoordinateEmbedding(erp_image_shape, index_counts=1)
        coordinate = CameraPrm(camera_angle=(phi, theta,), view_angle=(view_angle, view_angle),
                               image_plane_size=(image_shape, image_shape))
        coors.add_coordinate(coordinate, (1, image_shape, image_shape), 0)
        coor_erp,coor_mask = coors.get_coordinage()
        coor_erp = coor_erp.permute(0,3,1,2)
        extract = ImageExtraction(angle=angles,image_shape=extract_image_shape,view_angle=extract_view_angle,
                                  erp_image_shape=erp_image_shape)
        self.angle = angles
        self.extract_coor = extract(coor_erp,reshape=False).permute(0,2,3,1)
        self.extract_mask = extract(coor_mask,reshape=False)

    def __call__(self, img,reshape=True):
        print(self.extract_coor.shape,self.extract_mask.shape)
        extract = F.grid_sample(img, self.extract_coor.repeat(img.shape[0],1,1,1),mode="bilinear",align_corners=True)
        extract = extract * self.extract_mask
        mask = self.extract_mask
        if reshape:
            extract = rearrange(extract,"b c (t h) w->(b t) c h w",t=len(self.angle))
            mask = rearrange(mask,"b c (t h) w->(b t) c h w",t=len(self.angle))
        return extract,mask

if __name__ == '__main__':
    # imc1 = plt.imread("test.jpg")
    imc2 = plt.imread("test2.jpg")
    imc = np.stack([imc2,imc2],axis=0)
    extraction = ImageExtraction(view_angle=60)
    ext = extraction(torch.from_numpy(imc/255.).to(torch.float32).permute(0,3,1,2))

    coor_emb = CoordinateEmbedding((512,256))
    coor_emb.add_coordinate(camera_prm=CameraPrm(camera_angle=(0,0), view_angle=(45,45),image_plane_size=(256,256)),
                            inp_shape=(1,256,256))
    eic = ExtractImageConvertor(view_angle=60,cost_func_name="sup")
    recon = eic(ext)
    plt.imshow(recon[0].permute(1,2,0))
    plt.show()
    # plt.imshow(recon[1].permute(1,2,0))
    # plt.show()

    # emb = ImageEmbedding(0,80)
    # ext_mask = emb(ext[:1])
    # plt.imshow(ext_mask[0][0].permute(1, 2, 0))
    # plt.show()