
import os
import os.path as osp
import numpy as np
import torch
import cv2
from time import time
from PIL import Image, ImageFilter, ImageOps
import sys
# sys.path.append('..')
import facer


class FaceProcessor:
    def __init__(self, debug=False):
        self.root = osp.dirname(osp.abspath(__file__))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtct = facer.face_detector('retinaface/mobilenet', device=self.device)
        self.face_parser = facer.face_parser('farl/celebm/448', device=self.device) # optional "farl/lapa/448"
        self.face_aligner = facer.face_aligner('farl/ibug300w/448', device=self.device) # optional: "farl/wflw/448", "farl/aflw19/448" ibug300w
        self.debug = debug
        self.save_path = osp.join(self.root, 'inputs/deca_input')
        os.makedirs(self.save_path, exist_ok=True)
        self.area_dict = {'background':0, 'neck':1, 'face':2, 'cloth':3, 'rr':4, 'lr':5, 'rb':6, 'lb':7, 're':8, 'le':9, 'nose':10, 'imouth':11, 'llip':12, 'ulip':13, 'hair':14, 'eyeg':15, 'hat':16, 'earr':17, 'neck_l':18}
    def make_mask(self, parsing_anno, background=False):
        if background:
            parsing_anno = np.where(parsing_anno >= 100, 255, parsing_anno)
            parsing_anno = np.where(parsing_anno < 100, 0, parsing_anno)
        else:
            parsing_anno = np.where(parsing_anno >= 100, 200, parsing_anno)
            parsing_anno = np.where(parsing_anno < 100, 255, parsing_anno)
            parsing_anno = np.where(parsing_anno == 200, 0, parsing_anno)
        return parsing_anno
    
    def make_OBJ_mask(self, im, parsing_anno, mask_type, save_paths, save = False): # mask_type: 1 - neck, 2 - face, 3 - cloth, 4 - right_r, ..., 14 - hair 
        parsing_obj = parsing_anno['seg']['logits'][0][mask_type].cpu().numpy().astype(np.uint8)
        parsing_obj = self.make_mask( parsing_obj)
        face_mask = parsing_obj
        face_mask = np.where(face_mask <= 100, 0, face_mask)
        face_mask = np.where(face_mask >= 100, 255, face_mask)
        # face_mask = np.bitwise_not(face_mask)
        # cv2.imshow('face_mask', face_mask)
        im = im.astype(np.uint8)
        segmented_area = cv2.bitwise_and(im, im, mask=face_mask)
        if save: cv2.imwrite(save_path[:-4] +'face_mask.png', face_mask)
        return segmented_area, face_mask
    
    def make_OBJ_mask_multy(self, im, parsing_anno, mask_type_list, save_paths, save = False): # mask_type: 1 - neck, 2 - face, 3 - cloth, 4 - right_r, ..., 14 - hair 
        mask_ = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
        for mask_t in mask_type_list:
            mask_type = self.area_dict[mask_t]
            parsing_obj = parsing_anno['seg']['logits'][0][mask_type].cpu().numpy().astype(np.uint8)
            parsing_obj = self.make_mask( parsing_obj)
            face_mask = parsing_obj
            face_mask = np.where(face_mask <= 100, 0, face_mask)
            face_mask = np.where(face_mask >= 100, 255, face_mask)
            mask_ += face_mask
            # face_mask = np.bitwise_not(face_mask)
            # cv2.imshow('face_mask', face_mask)
            im = im.astype(np.uint8)
        segmented_area = cv2.bitwise_and(im, im, mask=mask_)
        if save: cv2.imwrite(save_path[:-4] +'face_mask.png', face_mask)
        return segmented_area, mask_
    
    def only_segment_face_area(self, im, parsing_anno, save_paths, save = False):
        parsing_r_ear = parsing_anno['seg']['logits'][0][4].cpu().numpy().astype(np.uint8)
        parsing_r_ear = self.make_mask(parsing_r_ear)
        parsing_l_ear = parsing_anno['seg']['logits'][0][5].cpu().numpy().astype(np.uint8)
        parsing_l_ear = self.make_mask( parsing_l_ear)
        parsing_hair = parsing_anno['seg']['logits'][0][14].cpu().numpy().astype(np.uint8)
        parsing_hair = self.make_mask( parsing_hair)
        parsing_background = parsing_anno['seg']['logits'][0][0].cpu().numpy().astype(np.uint8)
        parsing_background = self.make_mask( parsing_background, background=True)
        face_mask = parsing_background - parsing_hair - parsing_r_ear - parsing_l_ear #- parsing_ring 
        face_mask = np.where(face_mask <= 100, 0, face_mask)
        face_mask = np.where(face_mask >= 100, 255, face_mask)
        # cv2.imshow('face_mask', face_mask)
        im = im.astype(np.uint8)
        segmented_area = cv2.bitwise_and(im, im, mask=face_mask)
        if save: cv2.imwrite(save_path[:-4] +'face_mask.png', face_mask)
        return segmented_area, face_mask
#label_names ['background', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're', 'le', 'nose', 'imouth', 'llip', 'ulip', 'hair', 'eyeg', 'hat', 'earr', 'neck_l']
    
    def find_skin_color(self, im, save_path, parsing_anno, save = False):
        parsing_anno = parsing_anno['seg']['logits'][0][2].cpu().numpy().astype(np.int32)
        new_skin_mask = np.where(parsing_anno <= 1, 0, parsing_anno)
        if save: cv2.imwrite(save_path[:-4] +'skin_mmask.png', new_skin_mask)
        im = im.astype(np.uint8)
        new_skin_mask = new_skin_mask.astype(np.uint8)
        segmented_skin_area = cv2.bitwise_and(im, im, mask=new_skin_mask)
        mean_color_skin = cv2.mean(segmented_skin_area, mask=new_skin_mask)[:3]
        mean_color_skin = mean_color_skin[::-1]
        if self.debug: print('|------ mean_color_skin', mean_color_skin)
        if self.debug: print('|------ --------- mean_color_skin', mean_color_skin)
        if save: cv2.imwrite(save_path[:-4] +'skin.png', segmented_skin_area)
        return mean_color_skin, new_skin_mask, segmented_skin_area

    def paint_back(self, segmented_area, mean_color_skin, save_path):
        zeros_mask = np.all(segmented_area == [0, 0, 0], axis=-1)
        segmented_area[zeros_mask] = mean_color_skin
        segmented_area_back_painted = segmented_area.copy()
        return segmented_area_back_painted

    def hair_color(self, im, parsing_anno, save_path):
        new_hair_mask = np.where(parsing_anno != 17, 0, parsing_anno)
        new_hair_mask = np.where(new_hair_mask == 17, 255, new_hair_mask)
        segmented_hair_area = cv2.bitwise_and(im, im, mask=new_hair_mask)
        mean_color_hair = cv2.mean(segmented_hair_area, mask=new_hair_mask)[:3]
        return mean_color_hair, segmented_hair_area
    
    def create_gradient_mask(self, mask, width, sz):
        pil_mask = mask #Image.fromarray(mask)
        # pil_mask.show()
        mask = ImageOps.invert(pil_mask)
        mask = mask.filter(ImageFilter.MaxFilter(size=sz)) # 11
        mask = ImageOps.expand(mask, border=0, fill='black')
        gradient_mask = mask.copy()
        # width = 3
        gradient_mask = gradient_mask.filter(ImageFilter.GaussianBlur(width))
        return gradient_mask

    def blend_with_color(self, image, mask, color, width, sz, save=False):
        color = tuple([int(x) for x in color])
        color = (color[2], color[1], color[0], 255)
        color_image = Image.new(image.mode, image.size, color)
        gradient_mask = self.create_gradient_mask(mask, width, sz)
        if save: gradient_mask.save(self.save_path + '/_gradient_mask.png')
        color_image = color_image.crop((0, 0, image.width, image.height))
        gradient_mask = gradient_mask.resize(image.size)
        blended_image = Image.composite(color_image, image, gradient_mask)
        return blended_image

    def make_gradient_image(self, im, mean_color_skin, face_area_mask, width=5, sz=11):
        # cv2.imshow('face_area_mask', face_area_mask)
        # cv2.imshow('im', im)
        # cv2.waitKey(0)
        im_pil = Image.fromarray(im)
        mask = Image.fromarray(face_area_mask)
        segmented_areaGG = im_pil.convert('RGBA')
        blended_image = self.blend_with_color(segmented_areaGG, mask, mean_color_skin, width, sz)
        blended_image = blended_image.convert('RGB')
        blended_image = cv2.cvtColor(np.array(blended_image), cv2.COLOR_RGB2BGR)
        return blended_image
    
    def check_size(self, face_crop):
        height, width, _ = face_crop.shape
        if height == width:
            square_frame = face_crop
        else:
            if self.debug: print('----- not square\nperforming padding')
            max_dim = max(height, width)
            # square_frame = np.zeros((max_dim, max_dim, 3), dtype=face_crop.dtype)
            square_frame = np.ones((max_dim, max_dim, 3), dtype=face_crop.dtype) * 255
            y_margin = (max_dim - height) // 2
            x_margin = (max_dim - width) // 2
            square_frame[y_margin:y_margin+height, x_margin:x_margin+width, :] = face_crop
            # cv2.imwrite('inputs/padded_image.jpg', square_frame)
        return square_frame

    def facer_detect(self, img, out_path, vis=False, save=False):
        img = Image.fromarray(img)
        np_image = np.array(img.convert('RGB'))
        image =  facer.hwc2bchw(torch.from_numpy(np_image)).to(device=self.device)
        if self.debug: print('----- size of image', image.size())
        with torch.inference_mode():
            faces = self.dtct(image)
            if self.debug: print('----- faces', faces.keys())
            face_boxes = faces['rects']
            if self.debug: print('----- face_boxes', face_boxes.cpu().numpy())
        if vis:
            facer.show_bchw(facer.draw_bchw(image, faces))
        if save:
            out_img = facer.draw_bchw(image, faces)
            out_img = out_img.squeeze()
            if self.debug: print('----- shape of out_img', out_img.shape)
            out_path = out_path.replace('.jpg', '_det.jpg')
            self.write_hwc(out_img, out_path)
            if self.debug: print('----- saved to', out_path)
        return face_boxes, image, faces
    
    def facer_seg(self, img, faces, out_path, vis=False, save=False):
        if self.debug: print('----- face parsing on {} faces'.format(len(faces['rects'])))
        if self.debug: print('----- image shape', img.shape)
        with torch.inference_mode():
            faces = self.face_parser(img, faces)
        seg_logits = faces['seg']['logits']
        label_names = faces['seg']['label_names']
        if self.debug: print("----- keys for faces['seg']: ",faces['seg'].keys())
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        n_classes = seg_probs.size(1)
        seg_pr = seg_probs[0][14].cpu().numpy().astype(int)
        if self.debug: print('----- label_names', label_names)
        if self.debug: print('----- number of classes', n_classes)
        vis_seg_probs = seg_probs.argmax(dim=1).float()/n_classes*255
        vis_img = vis_seg_probs.sum(0, keepdim=True)
        if vis:
            facer.show_bchw(vis_img)
            facer.show_bchw(facer.draw_bchw(img, faces))
        if save:
            self.write_hwc(vis_img, out_path.replace('.jpg', '_mask.jpg'))
            self.write_hwc(facer.draw_bchw(img, faces), out_path.replace('.jpg', '_seg.jpg'))
            np.save(out_path.replace('.jpg', '_seg.npy'), seg_pr)

            seg_pr = np.where(seg_pr == 1, 255, seg_pr)
            cv2.imwrite(out_path.replace('.jpg', '_seg.png'), seg_pr)
        return faces    
    
    def facer_landmarks(self, image, faces, img, out_path, vis=False, save=False):
        with torch.inference_mode():
            faces = self.face_aligner(image, faces)
        landmarks = faces['alignment'].cpu().numpy().astype(np.int32)
        if vis:
            vis_img = img.copy()
            for points in landmarks:
                for point in points:
                    cv2.circle(vis_img, point, 1, (0, 255, 0), -1)
        if save:
            vis_img = img.copy()
            for points in landmarks:
                for point in points:
                    cv2.circle(vis_img, tuple(point), 1, (0, 255, 0), -1)
            cv2.imwrite(out_path.replace('.jpg', '_landmarks.jpg'), vis_img)
            # self.write_hwc(vis_img, out_path.replace('.jpg', '_mask.jpg'))

        
        
    def check_facer(self, img, out_path, vis=False, save=False):
        face_boxes, image, faces = self.facer_detect(img, out_path, vis, save=False)    
        self.facer_landmarks(image, faces, img, out_path, vis=False, save=True)
        if self.debug: print('----- done')
        return  faces
    
    def parse_face(self, img, out_path, vis=False, save=False):
        face_boxes, image, faces = self.facer_detect(img, out_path, vis, save=False)  
        print('----- faces', faces.keys())  
        faces = self.facer_seg( image, faces, out_path, vis=False, save=False)
        hair_mask = faces['seg']['logits'][0][14].cpu().numpy().astype(int)
        hair_mask = np.where(hair_mask <= 1, 255, hair_mask)
        return  faces    
    
    def parse_face2(self, img, out_path, vis=False, save=False):
        face_boxes, image, faces = self.facer_detect(img, out_path, vis, save=False)   
        print('----- faces', faces.keys())  
        print('----- faces', faces['rects'])
        print('----- faces len', len(faces['rects']))
        print('----- faces points', faces['points'])
        print('----- faces scores', faces['scores'])
        print('----- faces image_ids', faces['image_ids'])
        for detection in faces:
            bbox = faces['rects'][0].cpu().numpy()
            h, w, _ = img.shape
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
            margin = int(h / 2)
            x, y = max(0, x - margin), max(0, y - margin)
            w, h = int(w + 2 * margin), int(h + 2 * margin)
            new_img = img[y:y+h, x:x+w]
            new_bbox = [x, y, x+w, y+h]
            new_bbox_tensor = torch.tensor(new_bbox, dtype=torch.float32).cuda()
            faces['rects'] = [new_bbox_tensor.detach()]  # Use detach() to avoid in-place update
        print('----- faces rects new ', faces['rects'])
            # faces['rects'][0] = new_bbox_tensor.clone()  # Use clone() to avoid in-place update
            
            # if save_crop: cv2.imwrite(cropFileName, face)
                
        faces = self.facer_seg( image, faces, out_path, vis=False, save=False)
        hair_mask = faces['seg']['logits'][0][14].cpu().numpy().astype(int)
        hair_mask = np.where(hair_mask <= 1, 255, hair_mask)
        return  faces

    def write_hwc(self, image, path):
        image = image.squeeze()  # remove unnecessary dimensions
        if image.dim() == 2:  # if the image is grayscale
            image_np = image.cpu().numpy().astype(np.uint8)  # convert to uint8 numpy array
        elif image.dim() == 3:  # if the image is RGB
            image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        else:
            raise ValueError(f'Unsupported image shape: {image.shape}')
        Image.fromarray(image_np).save(path)  # create the PIL Image and save it

    def face_filter(self, input, save_path=None, save=False):
        save_path = save_path if save_path else self.save_path
        face = self.parse_face(input, save_path)
        mean_color_skin, new_skin_mask, segmented_skin_area = self.find_skin_color(im=input, save_path=save_path, parsing_anno=face)
        face_area_mask , msk= self.only_segment_face_area(im=input, parsing_anno=face, save_paths=save_path)
        # im_pil = Image.fromarray(input)
        if self.debug: print('----- mean_color_skin', mean_color_skin)
        if self.debug: print('----- msk', msk.shape)
        if self.debug: print('----- input', input.shape)
        if self.debug: print('type(msk)', type(msk))
        if self.debug: print('type(input)', type(input))
        if self.debug: print('data type(msk)', msk.dtype)
        if self.debug: print('data type(input)', input.dtype)
        # cv2.imshow('msk', msk)
        # cv2.imshow('input', input)
        # cv2.waitKey(0)
        blended_image = self.make_gradient_image(input, mean_color_skin, msk)
        clean_im_path = save_path.replace('.jpg', '_clean.jpg')
        if save: cv2.imwrite(clean_im_path, cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
        return blended_image, clean_im_path, mean_color_skin
    
    def face_obj_mask(self, input, save_path=None, save=False, vis=False):
        save_path = save_path if save_path else self.save_path
        face = self.parse_face(input, save_path)
        area_list = ['re', 'le','llip','hair']
        mask_list, sg_area_list = [], []
        for area in area_list:
            segmented_area, face_mask = self.make_OBJ_mask(input, face, self.area_dict[area], save_path, save = False)
            mask_list.append(face_mask)
            sg_area_list.append(segmented_area)
            if vis:
                cv2.imshow(f'face_mask for {area}', face_mask)
                cv2.imshow(f'segmented_area {area}', segmented_area)
                cv2.waitKey(0)
        return sg_area_list, mask_list, area_list
        #label_names ['background', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're', 'le', 'nose', 'imouth', 'llip', 'ulip', 'hair', 'eyeg', 'hat', 'earr', 'neck_l']
        # 0-background, 1-neck, 2-face, 3-cloth, 4-rr, 5-lr, 6-rb, 7-lb, 8-re, 9-le, 10-nose, 11-imouth, 12-llip, 13-ulip, 14-hair, 15-eyeg, 16-hat, 17-earr, 18-neck_l
    def face_obj_mask2(self, input, save_path=None, save=False, vis=False, area_name=None, area_list=None):
        save_path = save_path if save_path else self.save_path
        face = self.parse_face2(input, save_path)
 
        if area_list:
            segmented_area, face_mask = self.make_OBJ_mask_multy(input, face, area_list, save_path, save = False)

        else:
            segmented_area, face_mask = self.make_OBJ_mask(input, face, self.area_dict['hair'], save_path, save = False)
            mask_list.append(face_mask)
            sg_area_list.append(segmented_area)
        if vis:
            cv2.imshow(f'face_mask for {area_list[0]}', face_mask)
            cv2.imshow(f'segmented_area {area_list[0]}', segmented_area)
            cv2.waitKey(0)

        return segmented_area, face_mask, area_list
    
    
    
    def paint_to_custom_color(self, im, mask, color):
        im = im.copy()
        if self.debug: print('data type(im)', im.dtype)
        if self.debug: print('data type(mask)', mask.dtype)
        blended_image = self.make_gradient_image(im, color, mask)
        return blended_image
    
if __name__ == "__main__":
    t = time()
    root_dir = 'inputs/parsing'
    respth=f'{root_dir}/' 
    os.makedirs(respth, exist_ok=True)
    dspth = './inputs/celeb/celeb_a.jpg'
    dspth2 = './inputs/celeb/celeb_.jpg'
    folder = './inputs/celeb/'
    save_path=osp.join(respth, dspth.split("/")[-1])
    multy = False
    face_pr = FaceProcessor( debug=False)  
    only_mask = True 
    
    if multy:
        # face_pr = FaceProcessor( debug=False)  
        for file in os.listdir(folder):
            img_path = osp.join(folder, file)
            tt1 = time()
            print(f"processing {img_path}")
            img = cv2.imread(img_path)
            blended_image= face_pr.face_filter(input=img, save_path=save_path, save = True)
            rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
            cv2.imshow(f'blended_image_{file}', rgb)
            tt2 = time()
            print(f'time spend {tt2-tt1}')
        tt = time()
        print(f'total time spend {tt-t}')
    
    elif only_mask:
        t1 = time()
        print('----- start first round')
        img = cv2.imread(dspth)
        sg_area_list, mask_list, area_list = face_pr.face_obj_mask(img, save_path=save_path, save=False, vis=True) 
        
    
    else:
        t1 = time()
        print('----- start first round')
        # face_pr = FaceProcessor( debug=False)  
        img = cv2.imread(dspth)
        
        blended_image, clean_im_path, mean_color_skin = face_pr.face_filter(input=img, save_path=save_path, save = True)
        rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('blended_image', rgb)
        t2 = time()
        # print('----- start second round')
        # img = cv2.imread(dspth2)
        # blended_image, clean_im_path, mean_color_skin= face_pr.face_filter(input=img, save_path=save_path, save = True)
        # rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('blended_image2', rgb)
        # t3 = time()
        
        print(f'1st round time spend  = {t2-t1}')
        # print(f'2nd round time spend = {t3-t2}')
        # print(f'total time spent = {t3-t1}')



    cv2.waitKey(0)
    cv2.destroyAllWindows()
