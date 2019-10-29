#/usr/bin/env python
# -*- coding: utf-8 -*-

#import matplotlib
#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#import random
from csv import reader #, writer
#import scipy
#import time
#from itertools import chain, combinations
#import importlib
from pymcda.generate import generate_alternatives
from pymcda.pt_sorted import SortedPerformanceTable
from pymcda.generate import generate_random_performance_table
import random_model_generation_msjp
from pymcda.generate import generate_random_mrsort_model_msjp
from pymcda.generate import generate_random_performance_table_msjp
from random_model_generation_msjp import RandMRSortLearning 

import os
import glob


#DATADIR = os.getenv('DATADIR', '%s/python_workspace/pymcda-master/pymcda-data' % os.path.expanduser('~'))
#DATADIR = os.getenv('DATADIR', '%s/python_workspace/MRSort-jupyter-notebook' % os.path.expanduser('~'))
DATADIR = os.getenv('DATADIR')

class MRSortLearningResults():
    
    def __init__(self, directory, output_dir, nb_categories, nb_criteria, ticks_criteria, ticks_alternatives, \
                 nb_tests,nb_models, meta_l, meta_ll, meta_nb_models, noise = None):
        self.directory = directory
        self.nb_categories = nb_categories
        self.nb_criteria = nb_criteria
        self.ticks_criteria = ticks_criteria
        self.ticks_alternatives = ticks_alternatives
        self.nb_tests = nb_tests
        self.nb_models = nb_models
        self.meta_l = meta_l
        self.meta_ll = meta_ll
        self.meta_nb_models = meta_nb_models
        self.noise = noise
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
             os.makedirs(self.output_dir)
        if not os.path.exists(self.output_dir+"/computational_time"):
             os.makedirs(self.output_dir+"/computational_time")
        if not os.path.exists(self.output_dir+"/validation_CA"):
             os.makedirs(self.output_dir+"/validation_CA")
        if not os.path.exists(self.output_dir+"/tests_CA"):
             os.makedirs(self.output_dir+"/tests_CA")
        if not os.path.exists(self.output_dir+"/restitution"):
             os.makedirs(self.output_dir+"/restitution")


    def plot_boxplots(self, val, xlabels, ylabels, legend, save_file, labs, elem_plots, plot_type=3, marker="--", stat_fct = np.mean):
        if plot_type == 3:
            fig, ax = plt.subplots()
            #print(box_plot_data)
            box_plot_data = [[stat_fct(v) for v in val[x]] for x in val.keys()]
            if ylabels == "time (s)":
                ax.set_ylim([0.0,max([max(k) for k in box_plot_data])+1.0])
            else :
                ax.set_ylim([0,1.01])
            #import pdb; pdb.set_trace()
            lines = []
            #cm = plt.get_cmap('coolwarm')
            for i,dplot in zip(range(len(elem_plots)),box_plot_data):
                #lines += ax.plot(labs,dplot,marker,color=cm(i/len(elem_plots)))
                lines += ax.plot(labs,dplot,marker,color=plt.cm.hsv(i/len(elem_plots)))
            ax.set_xlabel(xlabels)
            ax.set_ylabel(ylabels)  
            ax.set_title(legend)
            ax.legend(lines,[str(i) for i in elem_plots])
            #plt.show()
            plt.savefig(save_file+"_all")
            plt.close()
        else:
            for i,x in zip(range(len(val.keys())),val.keys()):
                fig, ax = plt.subplots()
                #print(box_plot_data)
                if plot_type == 1:
                    box_plot_data = val[x]
                    ax.boxplot(box_plot_data,patch_artist=False,labels=labs,showmeans=True)
                    if ylabels == "time (s)":
                        ax.set_ylim([0.0,max([max(k) for k in val[x]])+1.0])
                    else :
                        ax.set_ylim([0.3,1.01])
                elif plot_type == 2:
                    box_plot_data = [stat_fct(v) for v in val[x]]
                    #print(x,val[x])
                    #print(labs)
                    #print(elem_plots)
                    #import pdb; pdb.set_trace()
                    if ylabels == "time (s)":
                        ax.set_ylim([0.0,max(box_plot_data)+1.0])
                    else :
                        ax.set_ylim([0.3,1.01])
                    ax.plot(labs,box_plot_data,marker)
                ax.set_xlabel(xlabels)
                ax.set_ylabel(ylabels)
                ax.set_title(legend+str(elem_plots[i]))
                #plt.show()
                plt.savefig(save_file+str(elem_plots[i]))
                plt.close()
    
    
    def plot_from_csv_results(self, dirs): #list of absolute path towards csv results files
        ca_validation = dict()
        ca_test = dict()
        comp_time = dict()
        rest_dupl_crit = dict()
        for x in dirs.keys():
            ca_validation[x] = []
            ca_test[x] = []
            comp_time[x] = []
            rest_dupl_crit[x] = []
            for file in dirs[x]:
                #import pdb; pdb.set_trace()
                file = max(glob.glob(file+"/plot*.csv"), key=os.path.getctime) 
                with open(file, 'r') as data_csv:
                    results = reader(data_csv,delimiter=";")
                    results= list(results)
                    #import pdb; pdb.set_trace()
                    comp_time[x] += [[float(i) for i in results[-4][0].split("[")[1].split("]")[0].split(",")]]
                    ca_validation[x] += [[float(i) for i in results[-3][0].split("[")[1].split("]")[0].split(",")]]
                    ca_test[x] += [[float(i) for i in results[-2][0].split("[")[1].split("]")[0].split(",")]]
                    #print(results[-1][0])
                    rest_dupl_crit[x] += [[float(i) for i in results[-1][0].split("[")[1].split("]")[0].split(",")]]
                #print(rest_dupl_crit)
        
        if list(dirs.keys()) == self.ticks_alternatives:
            elem_plots= list(dirs.keys())
            labs = [i.split("dupl")[-1] for i in dirs[elem_plots[0]]]
            #import pdb; pdb.set_trace()
            self.plot_boxplots(comp_time, "nb duplicated criteria", "time (s)","Avg computational time for na",self.output_dir + "/computational_time/time_dupl_crit_na",labs,elem_plots)
            self.plot_boxplots(ca_validation, "nb duplicated criteria", "% CA validation","Avg CA validation rate for na",self.output_dir + "/validation_CA/CA_validation_dupl_crit_na",labs, elem_plots)
            self.plot_boxplots(ca_test, "nb duplicated criteria", "% CA test","Avg CA test rate for na",self.output_dir + "/tests_CA/CA_test_dupl_crit_na",labs,elem_plots)
            self.plot_boxplots(rest_dupl_crit, "nb duplicated criteria", "% right criteria","% restitution for na", self.output_dir + "/restitution/restitution_dupl_crit_na",labs,elem_plots)

            self.plot_boxplots(comp_time, "nb duplicated criteria", "time (s)","Avg computational time for na",self.output_dir + "/computational_time/time_dupl_crit_na",labs,elem_plots,plot_type=2)
            self.plot_boxplots(ca_validation, "nb duplicated criteria", "% CA validation","Avg CA validation rate for na",self.output_dir + "/validation_CA/CA_validation_dupl_crit_na",labs, elem_plots,plot_type=2)
            self.plot_boxplots(ca_test, "nb duplicated criteria", "% CA test","Avg CA test rate for na",self.output_dir + "/tests_CA/CA_test_dupl_crit_na",labs,elem_plots,plot_type=2)
            self.plot_boxplots(rest_dupl_crit, "nb duplicated criteria", "% right criteria","% restitution for na", self.output_dir + "/restitution/restitution_dupl_crit_na",labs,elem_plots,plot_type=2)


        if list(dirs.keys()) == self.ticks_criteria:
            elem_plots = list(dirs.keys())
            labs = [i.split("na")[1].split("_")[0] for i in dirs[elem_plots[0]]]
            self.plot_boxplots(comp_time, "nb alternatives", "time (s)","Avg computational time for nb_dupl_crit",self.output_dir + "/computational_time/time_nb_alternatives_dupl",labs,elem_plots)
            self.plot_boxplots(ca_validation, "nb alternatives", "% CA validation","Avg CA validation rate for nb_dupl_crit",self.output_dir +"/validation_CA/CA_validation_nb_alternatives_dupl",labs,elem_plots)
            self.plot_boxplots(ca_test, "nb alternatives", "% CA test","Avg CA test rate for nb_dupl_crit",self.output_dir + "/tests_CA/CA_test_nb_alternatives_dupl",labs,elem_plots)
            self.plot_boxplots(rest_dupl_crit, "nb alternatives", "% right criteria","% restitution for nb_dupl_crit",self.output_dir+"/restitution/restitution_nb_alternatives_dupl",labs,elem_plots)

            self.plot_boxplots(comp_time, "nb alternatives", "time (s)","Avg computational time for nb_dupl_crit",self.output_dir + "/computational_time/time_nb_alternatives_dupl",labs,elem_plots,plot_type=2)
            self.plot_boxplots(ca_validation, "nb alternatives", "% CA validation","Avg CA validation rate for nb_dupl_crit",self.output_dir +"/validation_CA/CA_validation_nb_alternatives_dupl",labs,elem_plots,plot_type=2)
            self.plot_boxplots(ca_test, "nb alternatives", "% CA test","Avg CA test rate for nb_dupl_crit",self.output_dir + "/tests_CA/CA_test_nb_alternatives_dupl",labs,elem_plots,plot_type=2)
            self.plot_boxplots(rest_dupl_crit, "nb alternatives", "% right criteria","% restitution for nb_dupl_crit",self.output_dir+"/restitution/restitution_nb_alternatives_dupl",labs,elem_plots,plot_type=2)
            
        #return comp_time,ca_validation,ca_test,rest_dupl_crit
    

    def plot_all_results(self):
        str_noise = "_err" + str(self.noise) if self.noise != None else ""
        t = dict()
        for j in self.ticks_alternatives:
            t[j] = []
            for i in self.ticks_criteria:
                t[j] += [self.directory + "/rand_valid_test_na"+str(j)+"_nca" + str(self.nb_categories) + "_ncr"+str(nb_criteria)+"-0_dupl"+str(i)+str_noise]
        self.plot_from_csv_results(t)
        t = dict()
        for i in self.ticks_criteria:
            t[i] = []
            for j in self.ticks_alternatives:
                t[i] += [self.directory + "/rand_valid_test_na"+str(j)+"_nca" + str(self.nb_categories) + "_ncr"+str(nb_criteria)+"-0_dupl"+str(i)+str_noise]
        self.plot_from_csv_results(t)
        

    def exec_all_tests(self):
        for j in self.ticks_alternatives:
            for i in self.ticks_criteria:
                #nb_dupl_criteria = i
                print(" ... unit test nb_alternatives = %d, nb_duplicated_criteria = %d" %(j, i))
                self.l_dupl_criteria = list(range(self.nb_criteria))[:i]
                self.dir_criteria = [1] * (self.nb_criteria)
                inst = RandMRSortLearning(j, self.nb_categories, self.nb_criteria, self.dir_criteria, self.l_dupl_criteria, \
                            self.nb_tests, self.nb_models, self.meta_l, self.meta_ll, self.meta_nb_models, noise=self.noise)
                inst.learning_process()




######### Preparation of plot function calls
##############################################


if __name__ == "__main__":
    DATADIR = os.getenv('DATADIR')
    nb_categories = 2 #fixed
    nb_criteria = 6
    ticks_criteria = list(range(0,nb_criteria+1,2))
    ticks_alternatives = list(range(100,300,100))
    nb_tests = 10000
    nb_models = 10
    #Parameters of the metaheuristic MRSort
    # Parameters set by Sobrie for tests are : meta_l:30, meta_ll:20, meta_nb_models:10
    meta_l = 10
    meta_ll = 10
    meta_nb_models = 10
    directory = DATADIR
    output_dir = DATADIR + "/learning_results_plots"
    #test_instance = MRSortLearningResults(directory, output_dir, nb_categories, nb_criteria,ticks_criteria,ticks_alternatives, \
    #                nb_tests, nb_models, meta_l, meta_ll, meta_nb_models)
    #test_instance.exec_all_tests3()
    #test_instance.exec_all_tests()
    #test_instance.plot_all_results()



