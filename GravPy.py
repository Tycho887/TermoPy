 # -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:11:55 2022

@author: Michael
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import copy
import gc
import time

G = 6.6743e-11

class Object:
    
    def __init__(self,mass,position,velocity,name=''):
        self.name = name.strip()
        self.mass = mass
        self.pos = np.array(position)
        self.vel = np.array(velocity)
        self.pos_log = []
        self.energy = 0
        "Position and velocity are 3D vectors"
        
    def applyAcceleration(self,acc_vector,h):
        "h is the interval step length"
        #acc_vector = Force/self.mass
        self.vel += acc_vector*h
        #print(Force)
        self.pos += self.vel*h + 0.5*acc_vector*h**2
        self.pos_log.append(copy.deepcopy(self.pos))

    def __str__(self):
        return self.name

    def getPos(self):
        return self.pos_log

class System:
    def __init__(self,iterations,dt):
        self.iter = iterations
        self.dt = dt
        self.com_log = []
    def get_com(self,objects):
        for i in range(len(objects[0].pos_log)):
            com = sum([obj.mass*obj.pos_log[i] for obj in objects])
            total_mass = sum([obj.mass for obj in objects])
            self.com_log.append(com/total_mass)

            
    
    
def getObjectsFromFile(file='data/objects.csv'):
    objects = []
    with open(file,'r') as filein:
        #iterations,dt = filein.readline().split(';')
        for line in filein:
            name,mass,pos,vel = line.split(';')
            mass = float(mass)
            pos = list(map(float, pos.strip().split(',')))
            vel = list(map(float, vel.strip().split(',')))
            objects.append(Object(mass,pos,vel,name=name))
    return objects

def sortObjectsToMass(_objects):
    _objects.sort(key=lambda x: x.mass, reverse=True)
    return _objects

def gravity_equation(_object1):
    "Force from object1 on object2"
    "No need to multiply by mass of object 1"
    "This avoids a division operation per iteration"
    A = (G*_object1.mass)
    return A

def calculateAccelerationVectors(_objects, force_equation=gravity_equation):
    size = len(_objects)
    Acc_matrix = np.zeros(shape=(len(_objects[0].pos),size,size), dtype=float)
    for i in range(len(Acc_matrix[0,])):
        for j in range(i+1,len(Acc_matrix[0,])):
            
            vector = _objects[i].pos-_objects[j].pos
            distance = np.linalg.norm(vector)
            R2 = 1/distance**2
            unit_vector = vector/distance
            A1 = force_equation(_objects[i])*R2
            A2 = force_equation(_objects[j])*R2
            # print(F)
            Acc_matrix[:,i,j]=A1*unit_vector
            Acc_matrix[:,j,i]=-A2*unit_vector
#             print(f'''vector: {vector}
# distance: {distance}
# unit vector: {unit_vector}
# Acclereration: {A1,A2}''')

    "Calculates a matrix of every force between objects"
    "The resultant force on each object is given by the sum of forces along the"
    "1-axis of the 3-d matrix"
    "returns a matrix of forcevectors for each object"
    
    return np.sum(Acc_matrix, axis=1)

def ApplyAcceleration(_objects,h):
    Accelerations = calculateAccelerationVectors(_objects)
    for i in range(len(Accelerations[0,])):
        Acc = Accelerations[:,i]
        _objects[i].applyAcceleration(Acc,h)
        

def generate_data(_objects,iterations,dt):
    for i in range(iterations):
        print(f'Generating data {i}/{iterations}')
        ApplyAcceleration(_objects, dt)

def curate_data(_objects,iterations):
    for planet in _objects:
        planet.pos_log = planet.pos_log[::300]

def get_frame_size(objects):
    Max = []
    for i in objects:
        Max.append(np.max(i.pos_log))
    return max(Max)
        


def draw_frame(Objects):
    if len(objects[0].pos)==3:
        fig = plt.figure()
        axes = plt.subplot(111, projection='3d')
        for i in Objects:

            freq = len(i.pos_log) # finn en måte å moderere antall punkter som tegnes
            for j in i.pos_log:
                x, y, z = j[0],j[1],j[2]
                axes.plot(x, y, z, "o",color='blue')
            #ax.plot(i.pos[0],i.pos[1],i.pos[2], "o")
    else:
        for i in Objects:
            for j in i.pos_log:
                x,y = j[0],j[1]
                plt.scatter(x,y,s=1,color='blue')
            plt.scatter(i.pos[0],i.pos[1],s=50)
    plt.show()

#def center_of_mass


def system_energy(objects):
    pass
    # Finn den totale energien i systemet
        
def find_error(_objects):
    pass
    # Se på total energi før og etter for å finne optimal tidsintervall




if __name__ == "__main__":
    start = time.time()
    gc.collect()
    """
    Ettersom listen "objects" er en liste med objecter som selv
    inneholder deres egen posisjon, trengs det ikke 
    """
    h = 5; iterations = 100000
    objects = getObjectsFromFile('../data/earth_moon2d.csv')
    sortObjectsToMass(objects)
    generate_data(objects,iterations,h)
    curate_data(objects, iterations)
    print("Tegner frame")
    
    draw_frame(objects)
    
    print(f'Program runtime: {time.time()-start:.3f} seconds')
    