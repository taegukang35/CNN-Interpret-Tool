import pygame
import subprocess
from vgg16 import predict_class,get_featuremap,get_gradcam,get_lime

file_path = "dog.jpg"

# initialize game
pygame.init()
get_lime(file_path)

# screen option setting
size = [1700,1000]
screen = pygame.display.set_mode(size)
title = "CNN Visualization Tool"
pygame.display.set_caption(title)

# game setting
clock = pygame.time.Clock()

class obj:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.sx = 0
        self.sy = 0
    def put_img(self,img_path):
        if img_path[-3:] == 'png':
            self.img = pygame.image.load(img_path).convert_alpha()
        else:
            self.img = pygame.image.load(img_path)
        self.sx, self.sy = self.img.get_size()
    def change_size(self,sx,sy):
        self.img = pygame.transform.scale(self.img,(sx,sy))
        self.sx,self.sy = self.img.get_size()
    def show(self):
        screen.blit(self.img,(self.x,self.y))

## input object
inp = obj()
inp.put_img(file_path)
inp.change_size(200,200)
inp.x = 60
inp.y = 100

## vgg16 object
vgg16_struc = obj()
vgg16_struc.put_img("vgg16_struc.png")
vgg16_struc.x = round(size[0]/2-vgg16_struc.sx/2)
vgg16_struc.y = 10

## output object
myFont = pygame.font.SysFont("arial", 30, True, False)
pclass = myFont.render(predict_class(file_path)[0][1], True, (0, 0, 0))
pclass_rect = pclass.get_rect()
pclass_rect.centerx = 1550
pclass_rect.y = 160

## mode object
mode = pygame.font.SysFont("arial", 30, True, False)
mode = myFont.render("Mode: ", True, (0, 0, 0))
mode_rect = mode.get_rect()
mode_rect.centerx = 904+30
mode_rect.y = 407

## layer object
layer = pygame.font.SysFont("arial", 30, True, False)
layer = myFont.render("Layer: ", True, (0, 0, 0))
layer_rect = layer.get_rect()
layer_rect.centerx = 650-30
layer_rect.y = 407

## option object
option = obj()
option.put_img('option.PNG')
option.x = 290
option.y = 300

##filter object
filter = obj()
filter.x = 600
filter.y = 450


# main event
SB = 0
layer_name = 'block1_conv1'
option_name = 'feature'

def get_layer(mx,my):
    if 77<=my<=300:
        if 365<=mx<=403 :
            return 'block1_conv1'
        elif 405<=mx<=445 :
            return 'block1_conv2'
        elif 447<=mx<=484 :
            return 'block1_pool'
        elif 506<=mx<=548 :
            return 'block2_conv1'
        elif 550<=mx<=588 :
            return 'block2_conv2'
        elif 590<=mx<=628:
            return 'block2_pool'
        elif 652<=mx<=691 :
            return 'block3_conv1'
        elif 696<=mx<=732 :
            return 'block3_conv2'
        elif 735<=mx<=774 :
            return 'block3_conv3'
        elif 776<=mx<=816 :
            return 'block3_pool'
        elif 839<=mx<=878 :
            return 'block4_conv1'
        elif 881<=mx<=919 :
            return 'block4_conv2'
        elif 920<=mx<=960 :
            return 'block4_conv3'
        elif 962<=mx<=1002 :
            return 'block4_pool'
        elif 1024<=mx<=1064 :
            return 'block5_conv1'
        elif 1066<=mx<=1103 :
            return 'block5_conv2'
        elif 1108<=mx<=1146 :
            return 'block5_conv3'
        elif 1148<=mx<=1188 :
            return 'block5_pool'
        else: return -1
    else:
        return -1

def get_option(mx,my):
    if 326<=my<=370:
        if 310<=mx<=500:
            return 'feature'
        elif 530<=mx<=730:
            return 'filter'
        elif 760<=mx<=950:
            return 'grad'
        elif 980<=mx<=1175:
            return 'lime'
        elif 1205<=mx<=1400:
            return 'shap'
        else:
            return -1
    else:
        return -1

while SB == 0:
    # FPS setting
    clock.tick(1)
    screen.fill((255, 255, 255))

    # sense inputs
    for event in pygame.event.get():
        print(event)
        if event.type == pygame.QUIT:
            SB = 1
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx,my = pygame.mouse.get_pos()
            if get_layer(mx,my) != -1:
                layer_name = get_layer(mx,my)
            if get_option(mx,my) != -1:
                option_name = get_option(mx,my)
            print(layer_name,option_name)

    # change from inputs or time
    layer = myFont.render("Layer: " + layer_name, True, (0, 0, 0))

    if option_name == 'feature':
        get_featuremap(file_path,layer_name)
        feature = obj()
        feature.put_img('feature_map.png')
        feature.change_size(1500,500)
        feature.x = round(size[0]/2-feature.sx/2)
        feature.y = 450
        feature.show()
        mode = myFont.render("Mode: Feature Map", True, (0, 0, 0))
    elif option_name == 'grad':
        get_gradcam(file_path,layer_name)
        heatmap = obj()
        heatmap.put_img("heatmap.jpg")
        heatmap.change_size(400,400)
        heatmap.x = round(size[0]/2 - heatmap.sx/2-200)
        heatmap.y = 500
        grad_cam = obj()
        grad_cam.put_img("gradcam.jpg")
        grad_cam.change_size(400,400)
        grad_cam.x = round(size[0]/2+heatmap.sx/2-200)
        grad_cam.y = 500
        heatmap.show()
        grad_cam.show()
        mode = myFont.render("Mode: Grad-CAM", True, (0, 0, 0))
    elif option_name == 'filter':
        if 'conv' in layer_name:
            filter.put_img('vgg16_filter/'+layer_name+'.png')
            filter.change_size(500,500)
            filter.show()
            mode = myFont.render("Mode: Filter", True, (0, 0, 0))
    elif option_name == 'lime':
        layer = myFont.render("Layer: None", True, (0, 0, 0))
        lime = obj()
        lime_mask = obj()
        lime.put_img("lime.jpg")
        lime_mask.put_img('lime_mask.jpg')
        lime.change_size(400, 400)
        lime_mask.change_size(400,400)
        lime_mask.x = round(size[0] / 2 - lime_mask.sx / 2 - 200)
        lime_mask.y = 500
        lime.x = round(size[0]/2+lime.sx/2-200)
        lime.y = 500
        lime.show()
        lime_mask.show()
        mode = myFont.render("Mode: LIME", True, (0, 0, 0))
    elif option_name == 'shap':
        if 'conv' in layer_name:
            shap = obj()
            subprocess.call("python shap_vgg16.py --path " + file_path + " --layer_name " + layer_name)
            shap.put_img('shap.jpg')
            shap.x = round(size[0]/2-shap.sx/2)
            shap.y = 400
            shap.show()
            mode = myFont.render("Mode: SHAP", True, (0, 0, 0))
    # draw
    pygame.draw.rect(screen,(100,100,100),[1440,110,1658-1428,150])
    option.show()
    screen.blit(pclass,pclass_rect) #output 출력
    screen.blit(mode,mode_rect)
    screen.blit(layer,layer_rect)
    inp.show() #input 출력
    vgg16_struc.show() #vgg16 출력
    # update
    pygame.display.flip()
pygame.quit()
