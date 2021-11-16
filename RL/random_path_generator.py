import numpy as np
import math
from settings import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT

def random_path(tensor, width_of_path, num_crossings, cars=[]):
    # point of intersection.

    thetas=[]
    point_1=(np.random.randint(int(tensor.shape[1]*.5))+int(tensor.shape[1]*.25), np.random.randint(int(tensor.shape[2]*.5))+int(tensor.shape[2]*.25))

    tensor[0:2, point_1[0]-width_of_path:point_1[0]+width_of_path, point_1[1]-width_of_path: point_1[1]+width_of_path, 3] = 8 / NUM_SEM_CLASSES * np.ones((2, width_of_path*2, width_of_path*2))
    step=-1
    steps=[]
    for i in range(num_crossings):
        numbr=np.random.rand(1)

        theta=numbr
        thetas.append(theta)
        height_path=width_of_path
        if theta>0.5:
            theta = theta * math.pi
            #height_path=int(height_path/math.cos(math.pi-theta))
        else:
            theta = theta * math.pi
            #height_path = int(height_path / math.cos(theta))
        #height_path=min(height_path, 7)

        v1=1/math.tan(theta)

        if abs(numbr-0.5)>0.25:
            v1=-1/math.tan(math.pi-theta)
        if i+1>=num_crossings/2.0 and step<1:
            step=step*-1
        x_tmp = point_1[1]
        y_tmp = point_1[0]
        if num_crossings==1:
            if step>0:
                x_tmp=0
            else:
                x_tmp=tensor.shape[2]-1
            y_tmp = int((v1 * point_1[1] + point_1[0]) - (v1 * x_tmp))

        while x_tmp>0 and x_tmp<tensor.shape[2] and y_tmp>0 and y_tmp<tensor.shape[1]:
            tensor[0:2,y_tmp,x_tmp, 3]=8 / NUM_SEM_CLASSES*np.ones(2)
            if i<2:
                steps.append((y_tmp, x_tmp))
            if abs(numbr-0.5)>0.25:
                for x in range(x_tmp-height_path, x_tmp+height_path+1):
                    if x>=0 and x<tensor.shape[2]:
                        tensor[0:2, y_tmp,x, 3] = 8 / NUM_SEM_CLASSES*np.ones(2)
                if len(cars)>0:
                    cars[0].append([0,2, y_tmp,y_tmp+1,x_tmp+height_path+1,x_tmp+height_path*3/2+1])
                y_tmp = y_tmp + step
                x_tmp = int((point_1[0] - y_tmp) / v1 + point_1[1])
            else:
                for y in range(y_tmp-height_path, y_tmp+height_path+1):
                    if y>=0 and y<tensor.shape[1]:
                        tensor[0:2, y,x_tmp, 3] = 8 / NUM_SEM_CLASSES*np.ones(2)
                if len(cars)>0:
                    cars[0].append([0,2, y_tmp+height_path+1,y_tmp+height_path*3/2+1,x_tmp,x_tmp+1])
                x_tmp = x_tmp + step
                y_tmp = int((v1 * point_1[1] + point_1[0]) - (v1 * x_tmp))
            # if numbr > 0.45 and numbr < 0.65:
            #     x_tmp = x_tmp + step
            #     y_tmp = int((v1 * point_1[1] + point_1[0]) - (v1 * x_tmp))
            # else:
            #     y_tmp=y_tmp+step
            #     x_tmp=int((point_1[0]-y_tmp)/v1+point_1[1])
            #print str(x_tmp)+ " "+str(y_tmp)
    return tensor, point_1, thetas, steps, cars

def random_walk(tensor, width_of_path, num_steps,cars=[], horizontal=True, diagonal=False, dir=1, whitespace_width= -1):
    point_1 = [np.random.randint(int(tensor.shape[1]* .5)) + int(tensor.shape[1] * .25),
               np.random.randint(int(tensor.shape[2]* .5)) + int(tensor.shape[2] * .25)]
    positions=[]

    steps=np.random.randint(3,size=2*num_steps)


    if diagonal:
        if dir==1:
            point_1=[0,0]
        else:
            point_1 = [0,tensor.shape[2]-1]
    else:
        if horizontal:
            point_1[1] = 0
        else:
            point_1[0] = 0
    sz_t = tensor[0:2, point_1[0] - width_of_path:point_1[0] + width_of_path, point_1[1] - width_of_path: point_1[1] + width_of_path, 3].shape
    tensor[0:2, point_1[0] - width_of_path:point_1[0] + width_of_path, point_1[1] - width_of_path: point_1[1] + width_of_path, 3] = 8 / NUM_SEM_CLASSES * np.ones(
        sz_t)

    pos = point_1

    for step in range(num_steps):

        if diagonal:
            if step%2==0:

                pos[0] = pos[0] + width_of_path
                pos[1] = pos[1] + dir * width_of_path
            else:
                if step%4==1:
                    pos[0] = pos[0] + (steps[2 * step] - 1) * width_of_path

                else:
                    pos[1] = pos[1] + (steps[2 * step + 1] - 1) * width_of_path
        else:
            if horizontal:
                pos[0] = pos[0] + (steps[2 * step] - 1) * width_of_path
                pos[1] = pos[1] + width_of_path
            else:
                pos[0] = pos[0] + width_of_path
                pos[1] = pos[1] + (steps[2 * step + 1] - 1) * width_of_path
        if pos[0]>0 and pos[0]<tensor.shape[1]:
            if pos[1]>0 and pos[1]<tensor.shape[2]:
                positions.append(np.copy(pos))

        # sz_t = tensor[0:2, pos[0] - width_of_path*2:pos[0] - width_of_path, pos[1] - width_of_path*2:pos[1] + width_of_path*2,
        #        3].shape
        # if sz_t[1] > 0 and sz_t[2] > 0:
        if whitespace_width==-1:
            whitespace_width= width_of_path*2

        lower_lim_x = max(0, pos[0] - whitespace_width)
        upper_lim_x = min(tensor.shape[1],max(0, pos[0] + whitespace_width))
        lower_lim_y = max(0, pos[1] - whitespace_width)
        upper_lim_y = min(tensor.shape[2], pos[1] + whitespace_width)

        if upper_lim_y>lower_lim_y and upper_lim_x>lower_lim_x:
            indx=np.where(tensor[0, lower_lim_x:upper_lim_x,lower_lim_y:upper_lim_y,3]==11/NUM_SEM_CLASSES)

            for coord in zip(indx[0], indx[1]):
                tensor[0:2, pos[0] - width_of_path * 2+coord[0],pos[1] - width_of_path * 2+coord[1] ]=0
                #tensor[0:2, pos[0] - width_of_path * 2 + coord[0], pos[1] - width_of_path * 2 + coord[1]] = 0



            if len(indx)>0 and len(cars)>0 and pos[0] - width_of_path*3/2>0 and  pos[1]>0:
                cars[0].append([0, 2,pos[0] - width_of_path*3/2, pos[0] - width_of_path*3/2+2, pos[1], pos[1]+2])

        sz_t = tensor[0:2, pos[0] - width_of_path:pos[0] + width_of_path, pos[1] - width_of_path:pos[1] + width_of_path, 3].shape
        if sz_t[1]>0 and sz_t[2]>0:
            tensor[0:2, pos[0] - width_of_path:pos[0] + width_of_path, pos[1] - width_of_path:pos[1] + width_of_path, 3]= 8 / NUM_SEM_CLASSES * np.ones(sz_t)

            #loc=np.random.randint(tensor.shape[1]), np.random.randint(tensor.shape[2])
            #tensor[0:2,loc[0]-width_of_path: loc[0]+width_of_path, loc[1]-width_of_path: loc[1]+width_of_path, 3]=11/NUM_SEM_CLASSES*np.ones(tensor[0:2,loc[0]-width_of_path: loc[0]+width_of_path, loc[1]-width_of_path: loc[1]+width_of_path, 3].shape)
    return tensor, positions, cars
