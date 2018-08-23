import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np


def drop_from_list(list_in,drop_list):
    for xd in drop_list:
        if xd in list_in:
            list_in.pop(list_in.index(xd))
    return list_in

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
    
    
#https://www.esrl.noaa.gov/gmd/grad/neubrew/Calendar.jsp?view=DOY&year=2018&col=4
#Spring starts September 1 and ends November 30;
#Summer starts December 1 and ends February 28 (February 29 in a Leap Year);
#Fall   (autumn) starts March 1 and ends May 31; and
#Winter starts June 1 and ends August 31;
# Consideramos todos los años como bisiestos. Las Temporadas descriptas corresponden a las estaciones
# en el hemisferio sur.
def season_south_doy(x):
    if(x>=244 and x<=334):
        s='spring'
    if (x>60 and x<=151):
        s='fall'
    if (x>=152 and x<=243):
        s='winter'
    if (x>=335 or x<=60):
        s='summer'
    return s


def season_north_doy(x):
    if(x>=244 and x<=334):
        s='fall'
    if (x>60 and x<=151):
        s='spring'
    if (x>=152 and x<=243):
        s='summer'
    if (x>=335 or x<=60):
        s='winter'
    return s


def get_nearest_cent_mbound(pd_in,bounds,centroids):
    # centroids with same size as pd_in.cols
    pd_vals=pd_in.values# array
    sz_row=pd_vals.shape[0]; sz_col=pd_vals.shape[1]
    
    # Verificamos dimensionalidad de los constraints
    assert len(bounds)==sz_col 
    # Verificamos que estemos utilizando las mismas dimensiones en los features y los centroides
    assert centroids.shape[1]==sz_col 
    
    bounds_mat=np.tile(bounds,sz_row).reshape(sz_row,sz_col)
    ci_id=dict()
    ci_index=range(0,len(centroids))
    ci_label=[2*x+1 for x in ci_index]
        
    for ci,i,l in zip(centroids,ci_index,ci_label):
        ci_id[i]=[]
        aux_ci=[]
        ci_mat=np.tile(ci,sz_row).reshape(sz_row,sz_col)
        xc_diff=np.abs(pd_vals-ci_mat)
        scaled_bound=np.abs(np.multiply(ci_mat,bounds_mat))
        for j_row in range(0,sz_row):
            if all(xc_diff[j_row,:]<=scaled_bound[j_row,:]): # inside boundaries
                aux_ci.append(l)
                # Al identificar los puntos de esta forma nos aseguramos llevar la informaicon en el caso de una colision
                # es decir que se intente asignar el mismo punto en dos o mas clusters
            else:
                aux_ci.append(0)
        ci_id[i]=aux_ci
    pd_ci=pd.DataFrame(ci_id)
    pd_ci['merge']=pd_ci.sum(axis=1)
    return pd_ci,ci_id  




def get_nearest_cent_cosine(pd_in,bound,centroids):
    # centroids with same size as pd_in.cols
    pd_vals=pd_in.values# array
    sz_row=pd_vals.shape[0]; sz_col=pd_vals.shape[1]

    # Verificamos que estemos utilizando las mismas dimensiones en los features y los centroides
    assert centroids.shape[1]==sz_col 
    
    ci_id=dict()
    ci_index=range(0,len(centroids))
    ci_label=[2*x+1 for x in ci_index]
    
    for ci,i,l in zip(centroids,ci_index,ci_label):
        ci_id[i]=[]
        aux_ci=[]
        ci_mat=np.tile(ci,sz_row).reshape(sz_row,sz_col)
        for j_row in range(0,sz_row):
            x1=pd_vals[j_row,:];x2=ci_mat[j_row,:]
            xdot=np.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))    
            xdot=1 if xdot>1 else xdot
            xdot=-1 if xdot<-1 else xdot
            if np.abs(np.arccos(xdot))<bound: # inside boundaries
                aux_ci.append(l)
                # Al identificar los puntos de esta forma nos aseguramos llevar la informaicon en el caso de una colision
                # es decir que se intente asignar el mismo punto en dos o mas clusters
            else:
                aux_ci.append(0)
        ci_id[i]=aux_ci
    pd_ci=pd.DataFrame(ci_id)
    pd_ci['merge']=pd_ci.sum(axis=1)
    return pd_ci,ci_id  