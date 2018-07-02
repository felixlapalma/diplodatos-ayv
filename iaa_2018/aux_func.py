# Auxiliary

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
from ml.visualization import plot_confusion_matrix, classifier_boundary
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone

np.random.seed(1234)  # Setup seed to be more deterministic


def acc_model_par(model_spec,X_train_feature,y_train,X_val_feature,y_val,
                  var_target_name,var_target_vec,
                  var_target_name_aux,var_target_vec_aux):
    list_aux=[]
    p_dict={}
    for var,var_aux in zip(var_target_vec,var_target_vec_aux):
        model=clone(model_spec, safe=True)
        params_upd = {var_target_name:var}
        model.set_params(**params_upd)
        model.fit(X_train_feature, y_train)
        p_dict[var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux))]={}
        p_dict[var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux))].update({'model':model})
        acc_score_train=accuracy_score(y_train, model.predict(X_train_feature))
        acc_score_val=accuracy_score(y_val, model.predict(X_val_feature))
        # La exactitud toma valor en el rango [0, 1] donde más alto es mejor    
        p_dict[var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux))].update({'acc_score_train':acc_score_train,'acc_score_val':acc_score_val,var_target_name:var})
        list_aux.append([var_aux,var_target_vec_aux.index(var_aux),acc_score_train,acc_score_val])
    return p_dict,list_aux


def acc_model_two_par(model_spec,X_train_feature,y_train,X_val_feature,y_val,
                  var_target_name,var_target_vec,
                  var_target_name_aux,var_target_vec_aux):
    list_train_var=[]
    list_val_var=[]
    p_dict={}
    
    for var in var_target_vec:
        p_dict[var_target_name+'_idx_'+str(var_target_vec.index(var))]={}
        list_train_aux=[]
        list_val_aux=[]
        for var_aux in var_target_vec_aux:
            model=clone(model_spec, safe=True)
            params_upd = {var_target_name:var,var_target_name_aux:var_aux}
            model.set_params(**params_upd)
            model.fit(X_train_feature, y_train)
            #p_dict[var_target_name+'_idx_'+str(var_target_vec.index(var))].update({var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux)):{'model':model}})
            acc_score_train=accuracy_score(y_train, model.predict(X_train_feature))
            acc_score_val=accuracy_score(y_val, model.predict(X_val_feature))
            # La exactitud toma valor en el rango [0, 1] donde más alto es mejor    
            list_train_aux.append(acc_score_train)
            list_val_aux.append(acc_score_val)
            p_dict[var_target_name+'_idx_'+str(var_target_vec.index(var))].update({var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux)):{'acc_score_train':acc_score_train,'acc_score_val':acc_score_val,'model':model}})
        list_train_var.append(list_train_aux)
        list_val_var.append(list_val_aux)

    return p_dict,list_train_var,list_val_var



def mat_conf_plt(dataSet,model,X_train_feature,y_train,X_val_feature,y_val,title_add,figsize=(14,10),normNotNorm='both'):
    plt.figure(figsize=figsize, dpi= 80, facecolor='w', edgecolor='k')
    if normNotNorm=='both':
        plt.subplot(2, 2, 1)
        plot_confusion_matrix(confusion_matrix(y_train, model.predict(X_train_feature)),
                              classes=dataSet.target_names,
                              title='Matriz de confusión para entrenamiento (sin normalizar)'+title_add[0])
        plt.subplot(2, 2, 3)
        plot_confusion_matrix(confusion_matrix(y_train, model.predict(X_train_feature)),
                              classes=dataSet.target_names, normalize=True,
                              title='Matriz de confusión para entrenamiento (normalizado)'+title_add[1])

        plt.subplot(2, 2, 2)
        plot_confusion_matrix(confusion_matrix(y_val, model.predict(X_val_feature)),
                              classes=dataSet.target_names,
                              title='Matriz de confusión para validación (sin normalizar)'+title_add[2])
        plt.subplot(2, 2, 4)
        plot_confusion_matrix(confusion_matrix(y_val, model.predict(X_val_feature)),
                              classes=dataSet.target_names, normalize=True,
                              title='Matriz de confusión para validación (normalizado)'+title_add[3])
    elif normNotNorm=='norm':
        plt.subplot(1, 2, 1)
        plot_confusion_matrix(confusion_matrix(y_train, model.predict(X_train_feature)),
                              classes=dataSet.target_names, normalize=True,
                              title='Matriz de confusión para entrenamiento (normalizando)'+title_add[0])
        plt.subplot(1, 2, 2)
        plot_confusion_matrix(confusion_matrix(y_val, model.predict(X_val_feature)),
                              classes=dataSet.target_names, normalize=True,
                              title='Matriz de confusión para validación (normalizado)'+title_add[1])
    elif normoNotNorm=='not_norm':
        plt.subplot(1, 2, 1)
        plot_confusion_matrix(confusion_matrix(y_train, model.predict(X_train_feature)),
                              classes=dataSet.target_names, normalize=False,
                              title='Matriz de confusión para entrenamiento (sin normalizar)'+title_add[0])
        plt.subplot(1, 2, 2)
        plot_confusion_matrix(confusion_matrix(y_val, model.predict(X_val_feature)),
                              classes=dataSet.target_names, normalize=False,
                              title='Matriz de confusión para validación (sin normalizar)'+title_add[1])

 
    plt.show()

    
def boundary_plt(model,X_train_feature,X_val_feature,y_train,y_val,mesh_colors,classes_colors,title_add):
    plt.figure(figsize=(14, 5), dpi=80, facecolor='w', edgecolor='k')

    xx, yy, Z = classifier_boundary(np.r_[X_train_feature, X_val_feature], model)

    cmap_dots = classes_colors
    cmap_back = mesh_colors

    # Conjunto de entrenamiento
    plt.subplot(1, 2, 1)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_back)
    plt.scatter(X_train_feature[:, 0], X_train_feature[:, 1], c=y_train, cmap=cmap_dots, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Conjunto de Entrenamiento" + title_add[0])

    # Conjunto de validación
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_back)
    plt.scatter(X_val_feature[:, 0], X_val_feature[:, 1], c=y_val, cmap=cmap_dots, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Conjunto de Validación"+ title_add[1])

    plt.show()



def boundary_pol_plt(model,X_train_feature,X_val_feature,y_train,y_val,poly_features,mesh_colors,classes_colors,title_add):
    plt.figure(figsize=(14, 5), dpi=80, facecolor='w', edgecolor='k')

    xx, yy, Z = classifier_boundary(np.r_[X_train_feature, X_val_feature], model,poly_features)

    cmap_dots = classes_colors
    cmap_back = mesh_colors

    # Conjunto de entrenamiento
    plt.subplot(1, 2, 1)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_back)
    plt.scatter(X_train_feature[:, 0], X_train_feature[:, 1], c=y_train, cmap=cmap_dots, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Conjunto de Entrenamiento" + title_add[0])

    # Conjunto de validación
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_back)
    plt.scatter(X_val_feature[:, 0], X_val_feature[:, 1], c=y_val, cmap=cmap_dots, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Conjunto de Validación"+ title_add[1])

    plt.show()

def two_par_heatmap(matrix_train,matrix_val,xlabel,ylabel,title_list,figsize=(14,10)):
    plt.figure(figsize=(14, 10), dpi= 80, facecolor='w', edgecolor='k')
    plt.subplot(1, 2, 1)
    plt.imshow(matrix_train, interpolation='nearest')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_list[0])
    plt.colorbar() 
        
    plt.subplot(1, 2, 2)
    plt.imshow(matrix_val, interpolation='nearest')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_list[1])
    plt.colorbar() 
    plt.show()




def acc_model_pol_par(model_spec,X_train_feature,y_train,X_val_feature,y_val,
                  var_target_name,var_target_vec,
                  var_target_name_aux,var_target_vec_aux):
    list_train_var=[]
    list_val_var=[]
    p_dict={}
    
    for var in var_target_vec:
        p_dict[var_target_name+'_idx_'+str(var_target_vec.index(var))]={}
        list_train_aux=[]
        list_val_aux=[]
        poly_features = PolynomialFeatures(var)
        poly_features.fit(X_train_feature)
        X_poly_train = poly_features.transform(X_train_feature)
        X_poly_val = poly_features.transform(X_val_feature)
        for var_aux in var_target_vec_aux:
                
            model=clone(model_spec, safe=True)
            params_upd = {var_target_name_aux:var_aux}
            model.set_params(**params_upd)
            model.fit(X_poly_train, y_train)
            #p_dict[var_target_name+'_idx_'+str(var_target_vec.index(var))].update({var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux)):{'model':model}})
            acc_score_train=accuracy_score(y_train, model.predict(X_poly_train))
            acc_score_val=accuracy_score(y_val, model.predict(X_poly_val))
                    
            # La exactitud toma valor en el rango [0, 1] donde más alto es mejor    
            list_train_aux.append(acc_score_train)
            list_val_aux.append(acc_score_val)
            p_dict[var_target_name+'_idx_'+str(var_target_vec.index(var))].update({var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux)):{'acc_score_train':acc_score_train,'acc_score_val':acc_score_val,
                                               'model':model,
                                               'poly_features':poly_features}})
        list_train_var.append(list_train_aux)
        list_val_var.append(list_val_aux)

    return p_dict,list_train_var,list_val_var

def mse_model_pol_par(model_spec,X_train_feature,y_train,X_val_feature,y_val,
                  var_target_name,var_target_vec,
                  var_target_name_aux,var_target_vec_aux):
    list_train_var=[]
    list_val_var=[]
    p_dict={}
    
    for var in var_target_vec:
        p_dict[var_target_name+'_idx_'+str(var_target_vec.index(var))]={}
        list_train_aux=[]
        list_val_aux=[]
        poly_features = PolynomialFeatures(var)
        poly_features.fit(X_train_feature)
        X_poly_train = poly_features.transform(X_train_feature)
        X_poly_val = poly_features.transform(X_val_feature)
        for var_aux in var_target_vec_aux:
                
            model=clone(model_spec, safe=True)
            params_upd = {var_target_name_aux:var_aux}
            model.set_params(**params_upd)
            model.fit(X_poly_train, y_train)
            #p_dict[var_target_name+'_idx_'+str(var_target_vec.index(var))].update({var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux)):{'model':model}})
            mse_train=mean_squared_error(y_train, model.predict(X_poly_train))
            mse_val=mean_squared_error(y_val, model.predict(X_poly_val))        
            # La exactitud toma valor en el rango [0, 1] donde más alto es mejor    
            list_train_aux.append(mse_train)
            list_val_aux.append(mse_val)
            p_dict[var_target_name+'_idx_'+str(var_target_vec.index(var))].update({var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux)):{'mse_train':mse_train,'mse_val':mse_val,
                                               'model':model,
                                               'poly_features':poly_features}})
        list_train_var.append(list_train_aux)
        list_val_var.append(list_val_aux)

    return p_dict,list_train_var,list_val_var




def plt_reg_pol(model,poly_features,X_train_feature,X_val_feature,y_train,y_val,title_list_add):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')

    X_range_start = np.min(np.r_[X_train_feature, X_val_feature])
    X_range_stop = np.max(np.r_[X_train_feature, X_val_feature])
    y_range_start = np.min(np.r_[y_train, y_val])
    y_range_stop = np.max(np.r_[y_train, y_val])
    X_linspace = np.linspace(X_range_start, X_range_stop, 200).reshape(-1, 1)
    X_linspace_poly = poly_features.transform(X_linspace)
    
    # Conjunto de entrenamiento
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_feature, y_train, facecolor="dodgerblue", edgecolor="k", label="datos")
    plt.plot(X_linspace, model.predict(X_linspace_poly), color="tomato", label="modelo")
    plt.ylim(y_range_start, y_range_stop)
    title0="Conjunto de Entrenamiento" + '-' + title_list_add[0]
    plt.title(title0)
    
    # Conjunto de validación
    plt.subplot(1, 2, 2)
    plt.scatter(X_val_feature, y_val, facecolor="dodgerblue", edgecolor="k", label="datos")
    plt.plot(X_linspace, model.predict(X_linspace_poly), color="tomato", label="modelo")
    plt.ylim(y_range_start, y_range_stop)
    title0="Conjunto de Validación" + '-' + title_list_add[1]
    plt.title(title0)
    
    plt.show()



def plt_lin_set(model,X_train_feature,X_val_feature,y_train,y_val,title_list_add):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')

    X_range_start = np.min(np.r_[X_train_feature, X_val_feature])
    X_range_stop = np.max(np.r_[X_train_feature, X_val_feature])
    y_range_start = np.min(np.r_[y_train, y_val])
    y_range_stop = np.max(np.r_[y_train, y_val])
    X_linspace = np.linspace(X_range_start, X_range_stop, 200).reshape(-1, 1)
    
    
    # Conjunto de entrenamiento
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_feature, y_train, facecolor="dodgerblue", edgecolor="k", label="datos")
    plt.plot(X_linspace, model.predict(X_linspace), color="tomato", label="modelo")
    plt.ylim(y_range_start, y_range_stop)
    title0="Conjunto de Entrenamiento" + '-' + title_list_add[0]
    plt.title(title0)
    
    # Conjunto de validación
    plt.subplot(1, 2, 2)
    plt.scatter(X_val_feature, y_val, facecolor="dodgerblue", edgecolor="k", label="datos")
    plt.plot(X_linspace, model.predict(X_linspace), color="tomato", label="modelo")
    plt.ylim(y_range_start, y_range_stop)
    title0="Conjunto de Validación" + '-' + title_list_add[1]
    plt.title(title0)
    
    plt.show()





def mse_model_par(model_spec,X_train_feature,y_train,X_val_feature,y_val,
                  var_target_name,var_target_vec,
                  var_target_name_aux,var_target_vec_aux):
    list_aux=[]
    p_dict={}
    for var,var_aux in zip(var_target_vec,var_target_vec_aux):
        model=clone(model_spec, safe=True)
        params_upd = {var_target_name:var}
        model.set_params(**params_upd)
        model.fit(X_train_feature, y_train)
        p_dict[var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux))]={}
        p_dict[var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux))].update({'model':model})
        mse_train=mean_squared_error(y_train, model.predict(X_train_feature))
        mse_val=mean_squared_error(y_val, model.predict(X_val_feature))
        # La exactitud toma valor en el rango [0, 1] donde más alto es mejor    
        p_dict[var_target_name_aux+'_idx_'+str(var_target_vec_aux.index(var_aux))].update({'mse_train':mse_train,'mse_val':mse_val,var_target_name:var})
        list_aux.append([var_aux,var_target_vec_aux.index(var_aux),mse_train,mse_val])
    return p_dict,list_aux
