from math import ceil
from typing import Tuple

from scipy import ndimage
import torch
import numpy as np
from skimage.morphology import skeletonize, dilation
from scipy.ndimage import distance_transform_edt

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

# transform for radius map/internsity map, using for radius-aware Dice loss
# 半径/强度权重图变换，用于AGTAL
class WeightTransform(BasicTransform):
    def __init__(self, do_intensity=True, do_radius=False,spacing=None,alpha=0.5):
        """
        Calculates the radius/intensity of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "weight"
        """
        super().__init__()
        self.do_intensity=do_intensity
        self.do_radius=do_radius # enabled for radius map
        

        self.spacing=spacing
        self.alpha=alpha
    

    def apply(self, data_dict, **params):
        if self.do_radius and self.do_intensity:
            data_dict=self.apply_both(data_dict,alpha=self.alpha) # enabled for radius and internsity map 
        elif self.do_radius:
            data_dict=self.apply_radius(data_dict) # enabled for radius map
        elif self.do_intensity:
            data_dict=self.apply_intensity(data_dict) # enabled for internsity map
        return data_dict


    def distance_template(self,r,spacing):
        x,y,z=np.ogrid[-r:r+1,-r:r+1,-r:r+1]
        dis=np.sqrt((x*spacing[0])**2+(y*spacing[1])**2+(z*spacing[2])**2)
        return dis


    def apply_radius(self, data_dict):
        seg_all = data_dict['segmentation'].squeeze(0).numpy()

        
        # Add tubed skeleton GT
        bin_seg = np.zeros_like(seg_all, dtype=np.int16) # get binary label
        bin_seg[np.where(seg_all>0)]=1
        edt=distance_transform_edt(input=bin_seg, sampling=self.spacing) # calculate the distance between every artery label and nearest background
        radius=np.copy(edt).astype(np.float32)

        max_edt=int(np.max(edt)/np.min(self.spacing))+1 # get the maximum distance (cube distance)

        edt=np.pad(edt, ((max_edt,max_edt),(max_edt,max_edt),(max_edt,max_edt)), 'constant')

        distance_mask=self.distance_template(max_edt,self.spacing)



        for (i,j,k) in np.argwhere(bin_seg):
            roi=edt[i:i+max_edt*2+1,
                    j:j+max_edt*2+1,
                    k:k+max_edt*2+1]
        
            
            valid_mask=(distance_mask<=roi)
            valid_values=roi[valid_mask]

            if valid_values.size>0:
                radius[i,j,k]=np.unique(valid_values)[-1]
            


        max_radius=np.max(radius)
        
        # min_radius=np.unique(radius)[1]
        if np.count_nonzero(radius)==0:
            min_radius=0.469
        else:
            min_radius=np.min(radius[radius!=0])
        print('min_radius:',min_radius)

        
        
        weight_radius=(max_radius-radius)/(max_radius-min_radius)
        weight_radius=np.exp(weight_radius)

        weight_radius[np.where(radius==0)]=1

        
        data_dict["weight"] = torch.from_numpy(weight_radius).unsqueeze(0)

        return data_dict
    
    def apply_intensity(self, data_dict):
        seg_all = data_dict['segmentation'].numpy()

        origin_all=data_dict['image'].numpy()
        
        bin_seg = np.zeros_like(seg_all, dtype=np.int16)
        bin_seg[np.where(seg_all>0)]=1

        intensity_bin=origin_all*bin_seg

        max_int=np.max(intensity_bin)
        min_int=np.unique(intensity_bin)[1]

        intensity_all_weight=(max_int-intensity_bin)/(max_int-min_int)
        intensity_all_weight=np.exp(intensity_all_weight)


        intensity_all_weight[np.where(bin_seg==0)]=1 # the background must have the normal weight

        data_dict['weight']=torch.from_numpy(intensity_all_weight)

        return data_dict
    
    def apply_both(self, data_dict, alpha):
        data_dict_1=self.apply_radius(data_dict)

        radius_weight=data_dict_1['weight']

        data_dict_2=self.apply_intensity(data_dict)

        intensity_weight=data_dict_2['weight']

        weight=alpha*radius_weight + (1-alpha)*intensity_weight

        data_dict['weight']=weight

        return data_dict
    


class TopoWeightTransform(BasicTransform):
    def __init__(self, do_intensity=True, do_radius=False,spacing=None,alpha=0.01,lamada=20):
        """
        Calculates the skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        # self.do_tube = do_tube
        self.do_intensity=do_intensity
        self.do_radius=do_radius
        

        self.spacing=spacing
        self.alpha=alpha
        self.lamada=lamada
    

    def apply(self, data_dict, **params):
        if self.do_radius:
            data_dict=self.apply_radius(data_dict)
        

        return data_dict


    def distance_template(self,r,spacing):
        x,y,z=np.ogrid[-r:r+1,-r:r+1,-r:r+1]

        dis=np.sqrt((x*spacing[0])**2+(y*spacing[1])**2+(z*spacing[2])**2)
        return dis


    def apply_radius(self, data_dict):
        # seg_all = data_dict['segmentation'].numpy()
        seg_all = data_dict['segmentation'].squeeze(0).numpy()

        
        # Add tubed skeleton GT
        bin_seg = np.zeros_like(seg_all, dtype=np.int16)
        bin_seg[np.where(seg_all>0)]=1
        # print('bin_seg.shape:',bin_seg.shape)
        # bin_seg=bin_seg[0]
        # edt=distance_transform_edt(input=bin_seg, sampling=self.spacing)
        # radius=np.copy(edt).astype(np.float32)

        # max_edt=int(np.max(edt)/np.min(self.spacing))+1

        # edt=np.pad(edt, ((max_edt,max_edt),(max_edt,max_edt),(max_edt,max_edt)), 'constant')

        # distance_mask=self.distance_template(max_edt,self.spacing)
        # print('max_edt:',max_edt)

        # radius=np.zeros_like(seg_all, dtype=np.float32)
        

        # calculate the tube skeleton for avoid the edge point get the high response 
        
        
        # Skeletonize
        if not np.sum(bin_seg) == 0:
            skel = skeletonize(bin_seg)
            skel = (skel > 0).astype(np.int16)
            
            # skel = dilation(dilation(skel))
            # skel *= seg_all.astype(np.int16)
            # seg_all_skel = skel
            re_skel=np.ones_like(skel)
            re_skel[np.where(skel>0)]=0

            edge_edt, ind=distance_transform_edt(input=re_skel, sampling=self.spacing, return_indices=True)
            edge_seg=bin_seg*edge_edt

            max_edg=np.max(edge_seg)
            print(max_edg)

        weight_map=-1*self.lamada*np.log(edge_seg/max_edg+self.alpha)
        print(np.unique(weight_map)[-2])
            

        weight_map[np.where(bin_seg==0)]=1

        
        data_dict["weight"] = torch.from_numpy(weight_map).unsqueeze(0)
        # data_dict["weight"] = torch.from_numpy(weight_map)

        return data_dict
    

class RadiusDistanceTransform(BasicTransform):
    def __init__(self,spacing=None):
        """
        Calculates the radius of the label (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "radius"
        """
        super().__init__()
        self.spacing=spacing

    def distance_template(self,r,spacing):
        x,y,z=np.ogrid[-r:r+1,-r:r+1,-r:r+1]
        dis=np.sqrt((x*spacing[0])**2+(y*spacing[1])**2+(z*spacing[2])**2)
        return dis


    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].squeeze(0).numpy()



        # Add tubed skeleton GT
        bin_seg = np.zeros_like(seg_all, dtype=np.int16) # get binary label
        bin_seg[np.where(seg_all>0)]=1
        edt=distance_transform_edt(input=bin_seg, sampling=self.spacing) # calculate the distance between every artery label and nearest background
        radius=np.copy(edt).astype(np.float32)

        max_edt=int(np.max(edt)/np.min(self.spacing))+1 # get the maximum distance (cube distance)

        edt=np.pad(edt, ((max_edt,max_edt),(max_edt,max_edt),(max_edt,max_edt)), 'constant')

        distance_mask=self.distance_template(max_edt,self.spacing)



        for (i,j,k) in np.argwhere(bin_seg):
            roi=edt[i:i+max_edt*2+1,
                    j:j+max_edt*2+1,
                    k:k+max_edt*2+1]
        
            
            valid_mask=(distance_mask<=roi)
            valid_values=roi[valid_mask]

            if valid_values.size>0:
                radius[i,j,k]=np.unique(valid_values)[-1]
            


        max_radius=np.max(radius)
        
        if np.count_nonzero(radius)==0:
            min_radius=0.469
        else:
            min_radius=np.min(radius[radius!=0])



        weight_map=np.zeros_like(radius)

        # 自定义半径阈值
        for i,rad in enumerate([0.5,1.0,1.5,2.0,2.5]):
            if i==0:#(0~0.5,不包括0)
                weight_map[np.where((radius>0) & (radius<rad))]=i+1
            elif i==4:#>2.5
                weight_map[np.where((radius>=rad-0.5) & (radius<rad))]=i+1
                weight_map[np.where((radius>=rad))]=i+2
            else:
                weight_map[np.where((radius>=rad-0.5) & (radius<rad))]=i+1

        weight_map=weight_map.astype(np.int16)

        
        data_dict["radius"] = torch.from_numpy(weight_map).unsqueeze(0)

        return data_dict 
    
# 半径回归任务的半径标签变换
class RadiusRegTransform(BasicTransform):
    def __init__(self,spacing=None):
        """
        Calculates the radius of the label (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "radius"
        """
        super().__init__()
        

        self.spacing=spacing

    
    def distance_template(self,r,spacing):
        x,y,z=np.ogrid[-r:r+1,-r:r+1,-r:r+1]

        dis=np.sqrt((x*spacing[0])**2+(y*spacing[1])**2+(z*spacing[2])**2)
        return dis



    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].squeeze(0).numpy()

        
        # Add tubed skeleton GT
        bin_seg = np.zeros_like(seg_all, dtype=np.int16)
        bin_seg[np.where(seg_all>0)]=1
        # print('bin_seg.shape:',bin_seg.shape)
        # bin_seg=bin_seg[0]
        edt=distance_transform_edt(input=bin_seg, sampling=self.spacing)
        radius=np.copy(edt).astype(np.float32)

        max_edt=int(np.max(edt)/np.min(self.spacing))+1

        edt=np.pad(edt, ((max_edt,max_edt),(max_edt,max_edt),(max_edt,max_edt)), 'constant')

        distance_mask=self.distance_template(max_edt,self.spacing)
        # print('max_edt:',max_edt)

        # radius=np.zeros_like(seg_all, dtype=np.float32)
        

        # calculate the tube skeleton for avoid the edge point get the high response 
        
        
        # Skeletonize
        # if not np.sum(bin_seg) == 0:
        #     skel = skeletonize(bin_seg)
        #     skel = (skel > 0).astype(np.int16)
            
        #     skel = dilation(dilation(skel))
        #     # skel *= seg_all.astype(np.int16)
        #     # seg_all_skel = skel
        #     re_skel=np.ones_like(skel)
        #     re_skel[np.where(skel>0)]=0

        #     edge_edt, ind=distance_transform_edt(input=re_skel, sampling=self.spacing, return_indices=True)
        #     edge_seg=bin_seg*re_skel



        for (i,j,k) in np.argwhere(bin_seg):
            roi=edt[i:i+max_edt*2+1,
                    j:j+max_edt*2+1,
                    k:k+max_edt*2+1]
        
            
            valid_mask=(distance_mask<=roi)
            valid_values=roi[valid_mask]
            # print(np.unique(valid_values)[-1])

            if valid_values.size>0:
                radius[i,j,k]=np.unique(valid_values)[-1]
        

        
        data_dict["radius"] = torch.from_numpy(radius).unsqueeze(0)

        return data_dict 
    
# 中心线分类/回归任务变换
class CenterlineTransform(BasicTransform):
    def __init__(self,spacing=None,use_cla=False,cla_num=None):
        """
        Calculates the skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()

        self.use_cla=use_cla
        self.cla_num=cla_num
        self.spacing=spacing
    
    def distance_template(self,r,spacing):
        x,y,z=np.ogrid[-r:r+1,-r:r+1,-r:r+1]

        dis=np.sqrt((x*spacing[0])**2+(y*spacing[1])**2+(z*spacing[2])**2)
        return dis

    def apply(self, data_dict, **params):
        if self.use_cla:
            return self.apply_cla(data_dict, **params)
        else:
            return self.apply_reg(data_dict, **params)
        
    def apply_cla(self, data_dict, **params):
        seg_all = data_dict['segmentation'].squeeze(0).numpy()

        bin_seg=np.where(seg_all>0,1,0)
        heatmap=np.zeros_like(seg_all,dtype=np.float32)
        # Skeletonize
        if not np.sum(bin_seg) == 0:
            
            skel = skeletonize(bin_seg)
            heatmap[skel>0]=1
            tmp=np.copy(skel)
            # print('i:',0)
            # print(np.count_nonzero(tmp))
            if self.cla_num>1:
                for i in range(1,self.cla_num):
                    # print('i:',i)
                    tmp_2=dilation(tmp)
                    tmp_diff=tmp_2-tmp
                    tmp_diff=tmp_diff*bin_seg
                    # print(np.count_nonzero(tmp_diff))
                    heatmap[tmp_diff>0]=i
                    tmp=tmp_2

                tmp=(bin_seg-tmp)*bin_seg
                # print('cla:',self.cla_num)
                # print(np.count_nonzero(tmp))
                heatmap[tmp>0]=self.cla_num
            else:
                heatmap=dilation(tmp)
                heatmap=heatmap*bin_seg

        data_dict["radius"] = torch.from_numpy(heatmap).unsqueeze(0)


        return data_dict


    def apply_reg(self, data_dict, **params):
        seg_all = data_dict['segmentation'].squeeze(0).numpy()

        bin_seg=np.where(seg_all>0,1,0)
        heatmap=np.zeros_like(seg_all,dtype=np.float32)
                # Skeletonize
        if not np.sum(bin_seg) == 0:
            edge_edt=distance_transform_edt(input=bin_seg, sampling=self.spacing)
            
            skel = skeletonize(bin_seg)
            skel=skel.astype(np.int16)
            theta=edge_edt*skel

            
            
            sigma_list=np.unique(theta)[1:]
            # print('sigma_list:',sigma_list[::-1])
            for sigma in sigma_list[::-1]:
                mask=(skel>0) & (theta==sigma)
                mask_float=mask.astype(np.float32)

                smoothed=ndimage.gaussian_filter(mask_float,sigma=sigma)
                smoothed/=smoothed.max()
                

                heatmap=np.where(smoothed>heatmap,smoothed,heatmap)


        data_dict["radius"] = torch.from_numpy(heatmap).unsqueeze(0)

        return data_dict



#transform for keypoint map (using in the adjacency-aware co-occurrence loss-FN component)
# 关键点变换，用于AGTAL中的邻接关系共现损失
class KeyPointRegTransform(BasicTransform):
    def __init__(self,spacing=None) -> None:
        super().__init__()
        self.spacing=spacing
        self.key_point_list=[[1,2,3],#BA-L/RPCA1
                [1,15,16],# BA-L/R SCA
                [2,19,8],# RPCA1-RPcom-RPCA2
                [3,20,9],# LPCA1-LPcom-LPCA2
                [4,8],# RICA-RPcom
                [6,9],# LICA-LPcom
                [4,5,11],# RICA-RMCA-RACA1
                [6,7,12],# LICA-LMCA-LACA1
                [4,13],# RICA-RAChA
                [6,14],# LICA-LAChA
                [10,11,17],# Acom-RACA1-RACA2
                [10,12,18]]# Acom-LACA1-LACA2
    
    

    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].squeeze(0).numpy()
        bin_seg=np.where(seg_all>0,1,0)
        heatmap=np.zeros_like(seg_all,dtype=np.float32)

        if not np.sum(bin_seg) == 0:

            
            skel = skeletonize(bin_seg) # extract skeleton
            skel=skel.astype(np.int16)
            edge_edt=distance_transform_edt(input=bin_seg, sampling=self.spacing)
            
            key_points=np.zeros_like(seg_all)


            for value_ in self.key_point_list:
                key_array=np.zeros_like(seg_all)
                for va in value_:
                    key_array[np.where(seg_all==va)]=1 # current artery=1

                
                    for va_2 in value_:
                        if va==va_2:
                            continue
                        key_array[np.where(seg_all==va_2)]=10 # current artery's neighbor label=10
                
                    neigh_array=ndimage.maximum_filter(key_array,size=3,mode='constant') # if a voxel has a neighbor with 10, it will be 10
                    
                    AA=neigh_array==10 
                    BB=key_array==1 
                    tmp_=(AA & BB).astype(int) # find the voxels in current artery responsible for connection
                    key_points=np.where(tmp_>0,1,key_points) # these voxels is in the keypoint voxels

            key_points=dilation(dilation(key_points)) # dilate twice
            heatmap=key_points*bin_seg # keypoints map
            

        data_dict["keyP"] = torch.from_numpy(heatmap).unsqueeze(0)
        return data_dict
    





if __name__ == '__main__':
    import nibabel as nib
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Compute arterial metrics from TOF and markers.")
    parser.add_argument("origin_file", help="point file")
    parser.add_argument("label_file", help="point file")
    parser.add_argument("save_path", help="tof_file.")


    args=parser.parse_args()
    data_path=args.origin_file
    save_path=args.save_path
    label_file=args.label_file
    


    data=nib.load(label_file)
    label_img=data.get_fdata()

    data=nib.load(data_path)
    img=data.get_fdata()

    header=data.header
    spacing=header['pixdim'][1:4]
    print(spacing)

    img_torch=torch.from_numpy(img)

    label_img_torch=torch.from_numpy(label_img).unsqueeze(0)

    data_dict={'image': img_torch, 'segmentation': label_img_torch}

    trans_key=RadiusDistanceTransform(spacing=spacing)
    


    data_dict_int=trans_key.apply(data_dict=data_dict)
    weight=data_dict_int['radius'].squeeze(0).numpy()

    # save_file=os.path.join(save_path,'keyHeatmap.nii.gz')
    nib.save(nib.Nifti1Image(weight,data.affine,data.header),save_path)


