#/usr/bin/env python
# -*- coding: utf-8 -*-

#import matplotlib
#import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
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
                 nb_tests,nb_models, meta_l, meta_ll, meta_nb_models):
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
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
             os.makedirs(self.output_dir)


    def plot_boxplots(self, val, xlabels, ylabels, legend, save_file, labs):
        fig, ax = plt.subplots()
        box_plot_data = val
        if ylabels != "time (s)":
            ax.set_ylim([-0.1,1.10])
        else :
            ax.set_ylim([0.0,max([max(k) for k in val])+1.0])
        #print(box_plot_data)
        ax.boxplot(box_plot_data,patch_artist=False,labels=labs,showmeans=True)
        ax.set_xlabel(xlabels)
        ax.set_ylabel(ylabels)
        ax.set_title(legend)
        #plt.show()
        plt.savefig(save_file)
        plt.close()
    
    
    def plot_from_csv_results(self, dirs): #list of absolute path towards csv results files
        ca_validation = []
        ca_test = []
        comp_time = []
        rest_dupl_crit = []
        for file in dirs:
            #import pdb; pdb.set_trace()
            file = max(glob.glob(file+"/plot*.csv"), key=os.path.getctime) 
            with open(file, 'r') as data_csv:
                results = reader(data_csv,delimiter=";")
                results= list(results)
                #import pdb; pdb.set_trace()
                comp_time += [[float(i) for i in results[-4][0].split("[")[1].split("]")[0].split(",")]]
                ca_validation += [[float(i) for i in results[-3][0].split("[")[1].split("]")[0].split(",")]]
                ca_test += [[float(i) for i in results[-2][0].split("[")[1].split("]")[0].split(",")]]
                #print(results[-1][0])
                rest_dupl_crit += [[float(i) for i in results[-1][0].split("[")[1].split("]")[0].split(",")]]
                
        if len(dirs)>1 and dirs[0][-1] == dirs[1][-1]:
            dupl_crit = dirs[0][-1]
            labs = [i.split("na")[1].split("_")[0] for i in dirs]
            self.plot_boxplots(comp_time, "nb alternatives", "time (s)","Avg computational time for nb_dupl_crit"+dupl_crit,self.output_dir + "/time_nb_alternatives_dupl"+dupl_crit,labs)
            self.plot_boxplots(ca_validation, "nb alternatives", "% CA validation","Avg CA validation rate for nb_dupl_crit"+dupl_crit,self.output_dir + "/CA_validation_nb_alternatives_dupl"+dupl_crit,labs)
            self.plot_boxplots(ca_test, "nb alternatives", "% CA test","Avg CA test rate for nb_dupl_crit"+dupl_crit,self.output_dir + "/CA_test_nb_alternatives_dupl"+dupl_crit,labs)
            if self.l_dupl_criteria:
                self.plot_boxplots(rest_dupl_crit, "nb alternatives", "% right criteria","% restitution for nb_dupl_crit"+dupl_crit, self.output_dir + "/restitution_nb_alternatives_dupl"+dupl_crit,labs)
            
        if len(dirs)>1 and dirs[0][-1] != dirs[1][-1]:
            labs = [i[-1] for i in dirs]
            na = dirs[0].split("na")[1].split("_")[0]
            #import pdb; pdb.set_trace()
            self.plot_boxplots(comp_time, "nb duplicated criteria", "time (s)","Avg computational time for na"+na,self.output_dir + "/time_dupl_crit_na"+na,labs)
            self.plot_boxplots(ca_validation, "nb duplicated criteria", "% CA validation","Avg CA validation rate for na"+na,self.output_dir +"/CA_validation_dupl_crit_na"+na,labs)
            self.plot_boxplots(ca_test, "nb duplicated criteria", "% CA test","Avg CA test rate for na"+na,self.output_dir + "/CA_test_dupl_crit_na"+na,labs)
            if self.l_dupl_criteria:
                self.plot_boxplots(rest_dupl_crit, "nb duplicated criteria", "% right criteria","% restitution for na"+na,self.output_dir+"/restitution_dupl_crit_na"+na,labs)
        
        return comp_time,ca_validation,ca_test,rest_dupl_crit
    

    def plot_all_results(self):
        for j in self.ticks_alternatives:
            t = []
            for i in self.ticks_criteria:
                t += [self.directory + "/rand_valid_test_na"+str(j)+"_nca" + str(self.nb_categories) + "_ncr"+str(nb_criteria)+"-0_dupl"+str(i)]
            self.plot_from_csv_results(t)
    
        for i in self.ticks_criteria:
            t = []
            for j in self.ticks_alternatives:
                t += [self.directory + "/rand_valid_test_na"+str(j)+"_nca" + str(self.nb_categories) + "_ncr"+str(nb_criteria)+"-0_dupl"+str(i)]
            self.plot_from_csv_results(t)
        

    def exec_all_tests(self):
        for j in self.ticks_alternatives:
            for i in self.ticks_criteria:
                #nb_dupl_criteria = i
                print(" ... unit test nb_alternatives = %d, nb_duplicated_criteria = %d" %(j, i))
                self.l_dupl_criteria = list(range(self.nb_criteria))[:i]
                self.dir_criteria = [1] * (self.nb_criteria)
                inst = RandMRSortLearning(j, self.nb_categories, self.nb_criteria, self.dir_criteria, self.l_dupl_criteria, \
                            self.nb_tests, self.nb_models, self.meta_l, self.meta_ll, self.meta_nb_models)
                inst.learning_process()


    # version 2 where we consider the same model for all the tests
    def exec_all_tests2(self,classif_tolerance_prop = 0.10):
        self.dir_criteria = [1] * (self.nb_criteria)
        model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria)
        for j in self.ticks_alternatives:
            for i in self.ticks_criteria:
                print(" ... unit test nb_alternatives = %d, nb_duplicated_criteria = %d" %(j, i))
                self.l_dupl_criteria = list(range(self.nb_criteria))[:i]
                inst = RandMRSortLearning(j, self.nb_categories, self.nb_criteria, self.dir_criteria, self.l_dupl_criteria, \
                            self.nb_tests, self.nb_models, self.meta_l, self.meta_ll, self.meta_nb_models)
                inst.model = model
                inst.learning_process()


    # version 2 where we consider the same model for all the tests and add progressively alternatives..
    def exec_all_tests3(self):
        a=dict()
        pt=dict()
        aa=dict()
        learning_repr = 0
        nb_alternatives = 1
        self.dir_criteria = [1] * (self.nb_criteria)
        while (learning_repr > nb_alternatives*0.6) or (learning_repr < nb_alternatives*0.40) :
            model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria)
            for na in self.ticks_alternatives:
                nb_alternatives = na
                if na > 100 :
                    #import pdb; pdb.set_trace()
                    a[str(na)] = generate_alternatives(100,prefix ="a"+str(na)+"a" )
                    pt[str(na)] = generate_random_performance_table(a[str(na)], model.criteria)
                    a[str(na)].update(a[str(na - 100)])
                    pt[str(na)].update(pt[str(na - 100)])
                else :
                    a[str(na)] = generate_alternatives(na)
                    pt[str(na)] = generate_random_performance_table(a[str(na)], model.criteria)
                aa[str(na)] = model.get_assignments(pt[str(na)])
                learning_repr = len(aa[str(na)].get_alternatives_in_category('cat1'))
                if (learning_repr > nb_alternatives*0.6) or (learning_repr < nb_alternatives*0.40):
                    break

        for j in self.ticks_alternatives:
            for i in self.ticks_criteria:
                print(" ... unit test nb_alternatives = %d, nb_duplicated_criteria = %d" %(j, i))
                self.l_dupl_criteria = list(range(self.nb_criteria))[:i]
                inst = RandMRSortLearning(j, self.nb_categories, self.nb_criteria, self.dir_criteria, self.l_dupl_criteria, \
                            self.nb_tests, self.nb_models, self.meta_l, self.meta_ll, self.meta_nb_models)
                inst.model = model
                inst.a = a[str(j)]
                inst.pt = pt[str(j)]
                inst.prepare_copy_dupl_perftable()
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



