import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
import datetime
import random
import time
from collections import OrderedDict
import pprint
import matplotlib.pyplot as plt
import numpy as np
#from itertools import product
#from pymcda.learning.meta_mrsort3 import MetaMRSortPop3, MetaMRSortPop3AUC
#from pymcda.learning.meta_mrsortvc4 import MetaMRSortVCPop4
from pymcda.learning.meta_mrsortvc4_impl8 import MetaMRSortVCPop4MSJP
#from pymcda.learning.heur_mrsort_init_profiles import HeurMRSortInitProfiles
#from pymcda.learning.lp_mrsort_weights import LpMRSortWeights
#from pymcda.learning.lp_mrsort_veto_weights import LpMRSortVetoWeights
#from pymcda.learning.lp_mrsort_weights_auc import LpMRSortWeightsAUC
#from pymcda.learning.heur_mrsort_profiles4 import MetaMRSortProfiles4
#from pymcda.learning.heur_mrsort_profiles5 import MetaMRSortProfiles5
#from pymcda.learning.heur_mrsort_veto_profiles5 import MetaMRSortVetoProfiles5
#from pymcda.learning.lp_mrsort_mobius import LpMRSortMobius
#from pymcda.learning.heur_mrsort_profiles_choquet import MetaMRSortProfilesChoquet
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import Alternatives, Criteria, Criterion, PerformanceTable
from pymcda.types import AlternativePerformances
from pymcda.types import Criteria, Criterion
#from pymcda.types import AlternativesAssignments, Categories
#from pymcda.electre_tri import MRSort
from pymcda.generate import generate_alternatives
#from pymcda.generate import generate_categories_profiles
#from pymcda.generate import generate_random_profiles
#from pymcda.generate import generate_random_criteria_weights
from pymcda.pt_sorted import SortedPerformanceTable
#from pymcda.utils import compute_ca
#from pymcda.utils import compute_ca_good
#from pymcda.utils import compute_confusion_matrix
#from test_utils import test_result, test_results
#from test_utils import load_mcda_input_data
#from test_utils import save_to_xmcda
#from csv import reader, writer
#from pymcda.types import AlternativePerformances
#from pymcda.generate import generate_random_performance_table
#from pymcda.generate import generate_random_mrsort_model
from pymcda.generate import generate_random_mrsort_model_msjp
#from pymcda.generate import generate_random_mrsort_model_msjp_v3_explication
from pymcda.generate import generate_random_performance_table_msjp
from pymcda.generate import duplicate_performance_table_msjp
#from pymcda.utils import compute_winning_and_loosing_coalitions
from pymcda.utils import compute_confusion_matrix, print_confusion_matrix, add_errors_in_assignments
#import pdb
from copy import deepcopy
from collections import Counter

#DATADIR = os.getenv('DATADIR', '%s/python_workspace/Spyder/oso-pymcda/pymcda-data' % os.path.expanduser('~'))
##DATADIR = os.getenv('DATADIR', '%s/python_workspace/MRSort-jupyter-notebook' % os.path.expanduser('~'))
DATADIR = os.getenv('DATADIR')

# meta_mrsort = MetaMRSortVCPop4

class RandMRSortLearning():
    def __init__(self, nb_alternatives, nb_categories, nb_criteria, dir_criteria, l_dupl_criteria, nb_tests,\
                 nb_models, meta_l, meta_ll, meta_nb_models, noise = None, mu = 0.95, gamma = 0.5,\
                 renewal_method = 2, pretreatment = False, fixed_w1 = None, model_with_right_nb_crit = 0,\
                 model_heuristic = False, duplication = False, renewal_models = (0.5,0), strategy = (0,0), 
                 stopping_condition = 0, decision_rule = 1):
        self.nb_alternatives = nb_alternatives
        self.nb_categories = nb_categories
        self.nb_criteria = nb_criteria
        self.dir_criteria = dir_criteria
        self.nb_dupl_criteria = len(l_dupl_criteria)
        self.l_dupl_criteria = l_dupl_criteria
        self.nb_tests = nb_tests
        if pretreatment:
            self.meta_l = 1
            self.meta_ll = 1
            self.meta_nb_models = 1
        else :
            self.meta_l = meta_l
            self.meta_ll = meta_ll
            self.meta_nb_models = meta_nb_models
        if stopping_condition == 0:
            self.stopping_condition = self.meta_l
        else:
            self.stopping_condition = stopping_condition
        self.renewal_models = renewal_models
        self.strategy = strategy
        self.decision_rule = decision_rule
        self.nb_models = nb_models
        self.model = None
        self.pt = None
        self.noise = noise
        self.learned_models = []
        self.ca_avg = [0]*(self.nb_models)
        self.ca_tests_avg = [0]*(self.nb_models)
        self.ca_good_avg = 0
        self.ca_good_tests_avg = 0
        self.cpt_right_crit = dict()
        self.w_right_crit = [0]*(self.nb_models)
        self.w_dupl_right_crit = [0]*(self.nb_models)
        self.nb_null_weights = [0]*(self.nb_models)
        self.exec_time = [0]*(self.nb_models)
        self.cpt_gt_right_w = [0]*(self.nb_models) #count the number of time the good criteria got a greater weight than its duplicated criteria
        self.cpt_dupl_right_crit = dict()
        self.nb_under_lim_prof_val = [None]*(self.nb_models)
        self.nb_under_lim_prof_test = [None]*(self.nb_models)
        self.nb_heur_inc_positions = [None]*(self.nb_models)
        self.nb_heur_dec_positions = [None]*(self.nb_models)
        self.proportion_good_models = [None]*(self.nb_models)
        self.ca_iterations = [None]*(self.nb_models)
        self.ca_iterations0 = [None]*(self.nb_models)
        self.ca_iterations1 = [None]*(self.nb_models)
        self.ca_iterations3 = [None]*(self.nb_models)
        self.ca_iterations5 = [None]*(self.nb_models)
        self.ca_iterations7 = [None]*(self.nb_models)
        self.ca_iterations9 = [None]*(self.nb_models)
        self.ca_prefdir_iterations = [None]*(self.nb_models)
        self.ca_prefdir_iterations0 = [None]*(self.nb_models)
        self.ca_prefdir_iterations1 = [None]*(self.nb_models)
        self.ca_prefdir_iterations3 = [None]*(self.nb_models)
        self.ca_prefdir_iterations5 = [None]*(self.nb_models)
        self.ca_prefdir_iterations7 = [None]*(self.nb_models)
        self.ca_prefdir_iterations9 = [None]*(self.nb_models)
        self.ca_pop_avg_best = [None]*(self.nb_models)
        self.ca_pop_avg = [None]*(self.nb_models)
        self.end_iterations_meta = [0]*(self.meta_l)
        self.proportion_red_pts_end = [0]*(self.meta_nb_models+1)
        self.mu = mu
        self.gamma = gamma
        self.renewal_method = renewal_method
        self.pretreatment = pretreatment
        self.fixed_w1 = fixed_w1
        self.model_with_right_nb_crit = model_with_right_nb_crit
        self.model_heuristic = model_heuristic
        self.duplication = duplication
        self.stats_cav = 0
        self.stats_cag = 0
        self.stats_capd = 0
        self.stats_time = 0

        
    def generate_random_instance(self, classif_tolerance_prop = 0.10):
        b_inf = (self.nb_alternatives * 1.0 /self.nb_categories)-(classif_tolerance_prop*self.nb_alternatives)
        b_sup = (self.nb_alternatives * 1.0 /self.nb_categories)+(classif_tolerance_prop*self.nb_alternatives)
        notfound = True
        modelfound = False
        ptafound = False
        if self.model :
            modelfound = True
        if self.pt :
            ptafound = True
        while notfound :
            if not modelfound:
                #self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria, k=1)
                self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria, fixed_w1 = self.fixed_w1)
            if not ptafound:
                self.a = generate_alternatives(self.nb_alternatives)
                if self.duplication :
                    self.dupl_model_criteria = self.prepare_dupl_criteria_model()
                else :
                    self.dupl_model_criteria = self.model.criteria
                #self.pt,self.pt_dupl = generate_random_performance_table_msjp(self.a, self.model.criteria, dupl_crits = self.dupl_model_criteria, k=1)
                self.pt,self.pt_dupl = generate_random_performance_table_msjp(self.a, self.model.criteria, dupl_crits = self.dupl_model_criteria)
            self.aa = self.model.get_assignments(self.pt)
            i = 1
            size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
            while (size >= b_inf) and (size <= b_sup):
                if i == self.nb_categories:
                    notfound = False
                    break
                i += 1
                size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
        self.pt_dupl_sorted = SortedPerformanceTable(self.pt_dupl)
        if self.noise != None:
            self.aa_noisy = deepcopy(self.aa)
            self.aa_err_only = add_errors_in_assignments(self.aa_noisy, self.model.categories, self.noise)
        #self.pt_sorted = SortedPerformanceTable(self.pt)


    def prepare_dupl_criteria_model(self):
        #import pdb; pdb.set_trace()
        lcriteria = []
        for j in range(self.nb_criteria):
            if j in self.l_dupl_criteria:
                lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = 1, dupl_id=str(list(self.model.criteria.values())[j].id)+"d")]
            else :
                lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = 1)]
        lcriteria += [Criterion(id=str(j)+"d", direction = -1, dupl_id=j) for j in [list(self.model.criteria.values())[k].id for k in self.l_dupl_criteria]]
        #lnew_crit =  [self.model.criteria.values()[k].id for k in self.l_dupl_criteria]
        return Criteria(lcriteria)
        
    
    def prepare_copy_dupl_perftable(self):
        self.dupl_model_criteria = self.prepare_dupl_criteria_model()
        self.pt_dupl = duplicate_performance_table_msjp(self.pt, self.a, self.model.criteria, dupl_crits = self.dupl_model_criteria)
        #import pdb; pdb.set_trace()
              
    
    def heuristic_preference_directions(self, crit, prof_max_learned = 1, prof_min_learned  = 0):
        #import pdb; pdb.set_trace()
        prof_ord = self.pt_dupl_sorted.sorted_values[crit]
        #import pdb; pdb.set_trace()
        prof_ord = [(prof_ord[0]/2)] + [(prof_ord[i-1]+prof_ord[i])/2 for i in range(1,len(prof_ord))] + [(prof_ord[-1]+1)/2]
        #import pdb; pdb.set_trace()
        prof_max = {i : 0 for i in prof_ord}
        prof_min = {i : 0 for i in prof_ord}
        #import pdb; pdb.set_trace()
        for eb in prof_max.keys():
            for alt in self.a:
                #import pdb; pdb.set_trace()
                if self.aa(alt.id)=="cat2" and self.pt_dupl[alt.id].performances[crit] >= eb:
                    prof_max[eb] += 1
                if self.aa(alt.id)=="cat1" and self.pt_dupl[alt.id].performances[crit] < eb:
                    prof_max[eb] += 1
                if self.aa(alt.id)=="cat1" and self.pt_dupl[alt.id].performances[crit] > eb:
                    prof_min[eb] += 1
                if self.aa(alt.id)=="cat2" and self.pt_dupl[alt.id].performances[crit] <= eb:
                    prof_min[eb] += 1
        #import pdb; pdb.set_trace()
        prof_max = sorted(prof_max.values(),reverse = True)
        prof_min = sorted(prof_min.values(),reverse = True)
        #import pdb; pdb.set_trace()        
        return 1 if (prof_max>prof_min) else -1


    def plot_model(self, mod, mod2=None, iteration=0, num = 0, filename="", ca = 0, k=5):
        #import pdb; pdb.set_trace()
        fig, ax = plt.subplots()
        crits = list(mod.criteria.keys())
        #crits2 = list(mod2.criteria.keys())
        colors = ["black",None]
        if mod2:
            colors[1] = "blue" if (mod2.criteria[crits[0]].direction==1) else "red"
        lines = []
        thres =  round(mod.lbda,k)
        weights = [round(mod.cv[i].value,k) for i in crits]
        profiles = [mod.bpt['b1'].performances[i] for i in crits]
        xw = [str(crits[i])+"="+str(round(weights[i],k)) for i in range(len(crits))]
    
        if mod2:
            thres2 = round(mod2.lbda,k)
            weights2 = [round(mod2.cv[i].value,k) for i in crits]
            profiles2 = [mod2.bpt['b1'].performances[i] for i in crits]
            xw = [str(crits[i])+"="+str(round(weights2[i],k)) for i in range(len(crits))]
            
        lines += ax.plot(xw,profiles,color=colors[0])
        if mod2:
            lines += ax.plot(xw,profiles2,color=colors[1])
        

        #import pdb; pdb.set_trace()
        #print([str(crits[i])+"="+str(weights[i]) for i in range(len(crits))])
        #import pdb; pdb.set_trace()
        
        #ax.plot(range(len(crits)),profiles)
        #import pdb; pdb.set_trace()
        ax.set_ylim([-0.02,1.02])
        ax.set_xlabel("criteria")
        #plot.xticks([str(crits[i])+"="+str(weights[i]) for i in range(len(crits))])
        ax.set_ylabel("profile evaluation")
        
        if mod2:
            ax.set_title("Iter:" + str(iteration) + ", model" + str(num) + " : lbda="+str(thres2) + ", %ca="+str(ca))
            ax.legend(lines,["ori","model"+str(num)])
        else:
            ax.set_title("Original Model : lbda="+str(thres))
            ax.legend(lines,["original"])
        #ax.legend(loc="lower left")
        # add points min,max median values of alternatives evaluation on criteria w1
        
        for i,j in zip(range(len(crits)),crits):
            prof_evals = self.pt_dupl_sorted.sorted_values[j]
            plt.plot(i,np.median(prof_evals),".", color="green")
            plt.plot(i,np.min(prof_evals),".", color="green")
            plt.plot(i,np.max(prof_evals),".", color="green")
            plt.plot(i,np.percentile(prof_evals,25),".", color="green")
            plt.plot(i,np.percentile(prof_evals,75),".", color="green")
        #import pdb; pdb.set_trace()
        
        plt.grid()
        #plt.show()
        if filename:
            plt.savefig(filename)
        #plt.savefig("../pymcda-data/v32_explication0/iteration_"+str(i)+".png")
        plt.close()
        #import pdb; pdb.set_trace()



    def run_mrsort(self):
        categories = self.model.categories_profiles.to_categories()
#        self.proportion_good_models[self.num_model] = dict()
#        self.ca_iterations[self.num_model] = dict()
#        self.ca_iterations0[self.num_model] = dict()
#        self.ca_iterations1[self.num_model] = dict()
#        self.ca_iterations3[self.num_model] = dict()
#        self.ca_iterations5[self.num_model] = dict()
#        self.ca_iterations7[self.num_model] = dict()
#        self.ca_iterations9[self.num_model] = dict()
#        self.ca_prefdir_iterations[self.num_model] = dict()
#        self.ca_prefdir_iterations0[self.num_model] = dict()
#        self.ca_prefdir_iterations1[self.num_model] = dict()
#        self.ca_prefdir_iterations3[self.num_model] = dict()
#        self.ca_prefdir_iterations5[self.num_model] = dict()
#        self.ca_prefdir_iterations7[self.num_model] = dict()
#        self.ca_prefdir_iterations9[self.num_model] = dict()
#        self.ca_pop_avg[self.num_model] = dict()
#        self.ca_pop_avg_best[self.num_model] = dict()
        #percentile = 0
        fct_percentile = dict()
        fct_w_threshold = [0]*(self.meta_l)
        #fct_percentile[:15] = [(10,10)]*15
        for u in range(1,self.nb_dupl_criteria+1):
            fct_percentile["c"+str(u)] = [(0,0)]*(self.meta_l)
            if self.strategy[1] != 0:
                fct_percentile["c"+str(u)] = [(i,0) for i in np.arange(0,self.strategy[1],round((self.strategy[1])/self.meta_l,1))[::-1]]
            for i in range(len(fct_percentile["c"+str(u)])):
                fct_percentile["c"+str(u)][i] = (np.percentile(self.pt_dupl_sorted.sorted_values["c"+str(u)],fct_percentile["c"+str(u)][i][0]),np.percentile(self.pt_dupl_sorted.sorted_values["c"+str(u)],100-fct_percentile["c"+str(u)][i][0]))
        
        if self.strategy[0] != 0:
            fct_w_threshold = np.arange(0,self.strategy[0]+round((self.strategy[0]-0)/self.meta_l,3),round((self.strategy[0]-0)/self.meta_l,3))[::-1]
        #fct_w_threshold[:15] = [0.1]*15
        #print(len(fct_w_threshold),fct_w_threshold,fct_percentile,len(fct_percentile))
        
#        cacurr = 0
#        cacurr_nbit = 0
        if self.noise is None:
            #self.l_dupl_criteria = []
            #meta = MetaMRSortVCPop4MSJP(self.meta_nb_models, self.dupl_model_criteria, self.l_dupl_criteria, categories, self.pt_dupl_sorted, self.aa, gamma = self.gamma, renewal_method = self.renewal_method, duplication = True)
            meta = MetaMRSortVCPop4MSJP(self.meta_nb_models, self.dupl_model_criteria, self.l_dupl_criteria, \
                                        categories, self.pt_dupl_sorted, self.aa, gamma = self.gamma, \
                                        renewal_method = self.renewal_method, duplication = self.duplication, \
                                        fct_w_threshold = fct_w_threshold, fct_percentile = fct_percentile,\
                                        renewal_models = self.renewal_models, decision_rule = self.decision_rule)
        else:
            #meta = MetaMRSortVCPop4MSJP(self.meta_nb_models, self.dupl_model_criteria, self.l_dupl_criteria, categories, self.pt_dupl_sorted, self.aa_noisy, gamma = self.gamma, renewal_method = self.renewal_method, duplication = True)
            meta = MetaMRSortVCPop4MSJP(self.meta_nb_models, self.dupl_model_criteria, self.l_dupl_criteria, \
                                        categories, self.pt_dupl_sorted, self.aa_noisy, gamma = self.gamma, \
                                        renewal_method = self.renewal_method, duplication = self.duplication, \
                                        fct_w_threshold = fct_w_threshold, fct_percentile = fct_percentile)
        
        
        t1 = time.time()
        if self.num_model < 0:
            directory = DATADIR + self.output_dir+"V"+str(self.renewal_method)+"_details"+str(self.num_model)
            if not os.path.exists(directory):
                    os.makedirs(directory)
            
            filename = directory+"/model_original.png"
            self.plot_model(self.model,filename = filename)
            u=1
            while u <= self.nb_dupl_criteria:
                print("Model original, ",self.model.cv["c"+str(u)])
                u+=1
        for i in range(self.meta_l): #range(self.meta_l): #100 limit for another version
            #import pdb; pdb.set_trace()
            self.model2, ca, self.all_models, best_models = meta.optimize(self.meta_ll, 0, it_meta = i, cloning = False)
            #if i < 20:
#            hca = [h.ca for h in self.all_models]
#            self.ca_pop_avg[self.num_model][i] = np.mean(hca)
#            ind = int(np.ceil((1-np.mean(hca))/0.015))
#            if ind>=self.meta_nb_models:
#                ind = self.meta_nb_models
#                self.ca_pop_avg_best[self.num_model][i] = self.all_models[0].ca
#            else :
#                self.ca_pop_avg_best[self.num_model][i] = np.mean([h.ca for h in self.all_models][:(self.meta_nb_models-ind)])
            #import pdb; pdb.set_trace()
#            cacurr_nbit += 1
#            if cacurr != ca:
#                cacurr = ca
#                cacurr_nbit = 0
#            self.proportion_good_models[self.num_model][i] = [m.model.criteria["c1"].direction for m in self.all_models].count(1)
#            self.ca_iterations[self.num_model][i] = ca
#            self.ca_iterations0[self.num_model][i] = best_models[0].ca
#            self.ca_iterations1[self.num_model][i] = best_models[1].ca
#            self.ca_iterations3[self.num_model][i] = best_models[2].ca
#            self.ca_iterations5[self.num_model][i] = best_models[3].ca
#            self.ca_iterations7[self.num_model][i] = best_models[4].ca
#            self.ca_iterations9[self.num_model][i] = best_models[5].ca
#            # decision model 1:the best 2:the majority among 3,5,7,9
#            self.ca_prefdir_iterations[self.num_model][i] = 1 if (self.model2.criteria["c1"].direction==1) else 0
#            self.ca_prefdir_iterations0[self.num_model][i] = 1 if (best_models[0].model.criteria["c1"].direction==1) else 0
#            self.ca_prefdir_iterations1[self.num_model][i] = 1 if (best_models[1].model.criteria["c1"].direction==1) else 0
#            self.ca_prefdir_iterations3[self.num_model][i] = 1 if (best_models[2].model.criteria["c1"].direction==1) else 0
#            self.ca_prefdir_iterations5[self.num_model][i] = 1 if (best_models[3].model.criteria["c1"].direction==1) else 0
#            self.ca_prefdir_iterations7[self.num_model][i] = 1 if (best_models[4].model.criteria["c1"].direction==1) else 0
#            self.ca_prefdir_iterations9[self.num_model][i] = 1 if (best_models[5].model.criteria["c1"].direction==1) else 0
#            self.end_iterations_meta[i] += 1
            #import pdb; pdb.set_trace()
            #print("%d: ca: %f" % (i, ca))
            #print(self.model2.criteria)
            
            #b = [h.ca for h in self.all_models]
            #l = list(range(self.meta_nb_models))
            #ax.plot(l,b, marker="*")
            #print([h.model.criteria["c1"].direction for h in self.all_models])
            #colors = ["blue" if h.model.criteria["c1"].direction == 1 else "red" for h in self.all_models]
            if self.num_model < 0:
                scatter_crits = dict()
                for u in range(1,self.nb_dupl_criteria+1):
                    scatter_crits["c" + str(u)] = [[],[]]
                #scatter1 = []
                #scatter2 = []
                #print("Model original, ",self.model.cv["c1"])
                #import pdb; pdb.set_trace()
                
                #import pdb; pdb.set_trace()
                if not os.path.exists(directory+"/"+str(i)):
                    os.makedirs(directory+"/"+str(i))
                for j,h in zip(range(self.meta_nb_models),self.all_models):
                    filename = directory+"/"+str(i)+"/model" + str(j) + ".png"
                    self.plot_model(self.model, h.model, iteration=i, num=j, filename = filename, ca = round(h.ca,3))
                    #self.plot_model(self.model)
                    #import pdb; pdb.set_trace()
                    for u in range(self.nb_dupl_criteria):
                        if h.model.criteria["c"+str(u+1)].direction == 1:
                            scatter_crits["c" + str(u+1)][0] += [j+(u/(2*self.nb_dupl_criteria)),h.ca]
                        if h.model.criteria["c"+str(u+1)].direction == -1:
                            #print(j,h.model.cv["c1"],h.model.bpt['b1'].performances["c1"])
                            scatter_crits["c" + str(u+1)][1] += [j+(u/(2*self.nb_dupl_criteria)),h.ca]
                
                fig, ax = plt.subplots()
                #print(scatter1[1::2],scatter2[1::2])
                #print([[x,y] for x,y in zip(scatter1[1::2],scatter2[1::2])])
                markers = [".","*","+","^","°"]
                for u in range(self.nb_dupl_criteria):
                    ax.scatter(scatter_crits["c" + str(u+1)][0][0::2],scatter_crits["c" + str(u+1)][0][1::2], marker=markers[u], c="blue",label="dir inc c"+str(u+1))
                    ax.scatter(scatter_crits["c" + str(u+1)][1][0::2],scatter_crits["c" + str(u+1)][1][1::2], marker=markers[u], c="red",label="dir dec c"+str(u+1))
                
                #b = [ , , , ,0.27]
                #ax.plot(l,b)
                ax.set_ylim([0.5,1.01])
                oo = round(np.mean([h.ca for h in self.all_models]),3)
                ax.set_xlabel("model number")
                if self.meta_nb_models > 1:
                    plt.xticks(range(0,self.meta_nb_models,10))
                else:
                    plt.xticks(range(0,self.meta_nb_models,1))
                ax.set_ylabel("% CA")
                ax.set_title("Iter:" + str(i) + ", %CA of models, w1_ori="+str(self.model.cv["c1"].value) + " %avg="+ str(oo) + " new=" + str(int(((1-oo)/0.15)*self.meta_nb_models)))
                ax.legend(loc="lower left")
                plt.minorticks_on()
                #plt.grid()
                #plt.show()
                plt.savefig(directory+"/"+str(i)+"/models_population_it"+str(i)+".png")
                plt.close()
                #import pdb; pdb.set_trace()
                #constraint to stop the number of iteration when 
#                if ca == 1:
#                    break
#                if cacurr_nbit >= 30 and [h.ca for h in self.all_models].count(cacurr) >= 3:
#                    break
        #print(self.num_model,self.model2.criteria["c1"])
        t2 = time.time()
#        self.proportion_red_pts_end[[m.model.criteria["c1"].direction for m in self.all_models].count(-1)] += 1
        #import pdb; pdb.set_trace()
        #self.nb_heur_dec_positions[self.num_model] = self.model2.cv["c1"].value
        
        #print([h.model.criteria["c1"].direction for h in self.all_models])
        #print([h.model.criteria["c2"].direction for h in self.all_models])
        #print([h.model.criteria["c3"].direction for h in self.all_models])
        #print([h.model.criteria["c4"].direction for h in self.all_models])
        #print([h.model.criteria["c5"].direction for h in self.all_models])
#        if not self.duplication :
#            l_opt = [0]* len(self.l_dupl_criteria)
#            #import pdb; pdb.set_trace()
#            for g in self.l_dupl_criteria:
#                #print([list(h.model.criteria.values())[g].direction for h in self.all_models])
#                #print([list(h.model.criteria.values())[g].direction for h in self.all_models][:self.meta_nb_models//2])
#                tmpr =[list(h.model.criteria.values())[g].direction for h in self.all_models][:self.meta_nb_models//2].count(1)
#                if tmpr > self.meta_nb_models//4:
#                    l_opt[g] = 1
#                else :
#                    l_opt[g] = -1
#            #print(tmpr,l_opt)
#            #import pdb; pdb.set_trace()
#            y=0
#            mmax = 0
#            while y < self.meta_nb_models//2:
#                #print(list(self.all_models[:self.meta_nb_models//2][::-1][y].model.criteria.values())[0].direction)
#                #print(l_opt[0])
#                #print([k for k in self.l_dupl_criteria])
#                #import pdb; pdb.set_trace()
#                tmpm = [1 for k in self.l_dupl_criteria if list(self.all_models[:self.meta_nb_models//2][::-1][y].model.criteria.values())[k].direction == l_opt[k]]
#                #print(y,tmpm)
#                if mmax <= sum(tmpm):
#                    self.model2 = self.all_models[:self.meta_nb_models//2][::-1][y].model
#                    mmax = sum(tmpm)
#                y += 1
#            #print(self.model2.criteria)
#            #print([h.model.criteria["c1"].direction for h in self.all_models])
#            #print([h.model.criteria["c2"].direction for h in self.all_models])
#            #print([h.model.criteria["c3"].direction for h in self.all_models])
#            #print([h.model.criteria["c4"].direction for h in self.all_models])
#            #print([h.model.criteria["c5"].direction for h in self.all_models])
#            #import pdb; pdb.set_trace()
        self.exec_time[self.num_model] = (t2-t1)
        #print(self.model2.cv)
        #import pdb; pdb.set_trace()
        return self.exec_time[self.num_model]


    def eval_model_validation(self):
        self.nb_under_lim_prof_val[self.num_model]=dict()
        self.aa_learned = self.model2.get_assignments(self.pt_dupl)
        total = len(self.a)
        nok = 0
        totalg = 0
        okg = 0
        ref_crit = [i for i,j in self.model2.criteria.items()]
        tab_max = dict()
        tab_min = dict()
        for alt in self.a:
            if self.aa(alt.id) != self.aa_learned(alt.id):
                nok += 1
            if self.aa(alt.id) == "cat1":
                totalg += 1
                if self.aa_learned(alt.id) == "cat1":
                    okg += 1
            #import pdb; pdb.set_trace()
            #tmpi = 0
            if self.model_with_right_nb_crit == 0: #preference direction restoration rate
                for i,j in self.model2.criteria.items():
                    if i[-1] == "d":
                        #import pdb; pdb.set_trace()
                        #print(list(self.model2.cv.values()))
                        cmax = float(list(self.model2.cv.values())[ref_crit.index(j.dupl_id)].value)
                        cmin = float(list(self.model2.cv.values())[ref_crit.index(i)].value)
                        if cmax != 0 and cmin != 0 :
                            #import pdb; pdb.set_trace()
                            if not j.dupl_id in tab_max.keys():
                                tab_max[j.dupl_id] = 0
                                tab_min[j.dupl_id] = 0
                            if self.model2.bpt['b1'].performances[j.dupl_id] < self.pt_dupl[alt.id].performances[j.dupl_id]:
                                tab_max[j.dupl_id] += 1           
                            if self.model2.bpt['b1'].performances[i] < self.pt_dupl[alt.id].performances[i]:
                                tab_min[j.dupl_id] += 1
                        if cmax == 0 and cmin == 0 :
                            if not j.dupl_id in tab_max.keys():
                                tab_max[j.dupl_id] = 0
                                tab_min[j.dupl_id] = 0
        #print(cmax,cmin,tab_max,tab_min,self.model2.bpt['b1'].performances[j.dupl_id],self.model2.bpt['b1'].performances[i])
        #import pdb; pdb.set_trace()
        if self.model_with_right_nb_crit == 0: # S1
            for i in tab_min.keys():
                if (max(tab_max[i],self.nb_alternatives-tab_max[i]) < self.mu*self.nb_alternatives) and (max(tab_min[i],self.nb_alternatives-tab_min[i]) > self.mu*self.nb_alternatives):
                    self.nb_under_lim_prof_val[self.num_model][i] = 1
                elif (max(tab_max[i],self.nb_alternatives-tab_max[i]) > self.mu*self.nb_alternatives) and (max(tab_min[i],self.nb_alternatives-tab_min[i]) < self.mu*self.nb_alternatives):
                    self.nb_under_lim_prof_val[self.num_model][i] = 0
                else:
                    #import pdb; pdb.set_trace()
                    if self.heuristic_preference_directions(i) == 1:
                        self.nb_under_lim_prof_val[self.num_model][i] = 1
                    else :
                        self.nb_under_lim_prof_val[self.num_model][i] = 0

        if not self.duplication: # S4
            self.nb_under_lim_prof_val[self.num_model]["c1"] = self.nb_dupl_criteria
            for i,j in self.model2.criteria.items():
                if j.direction == -1:
                    self.nb_under_lim_prof_val[self.num_model]["c1"] -= 1
                #[h.model.criteria[i].direction for h in self.all_models]
        
        #print(self.nb_under_lim_prof_val[self.num_model])       
        #import pdb; pdb.set_trace()
        totalg = 1 if totalg == 0 else totalg
        self.ca_avg[self.num_model] = float(total-nok)/total
        self.ca_good_avg += (float(okg)/totalg)

        return (float(total-nok)/total),(float(okg)/totalg)


    def eval_model_test(self):
        self.nb_under_lim_prof_test[self.num_model]=dict()
        a_tests = generate_alternatives(self.nb_tests)
        #pt_tests = generate_random_performance_table(a_tests, self.dupl_model_criteria)
        pt_tests,pt_tests_dupl = generate_random_performance_table_msjp(a_tests, self.model.criteria, dupl_crits = self.dupl_model_criteria)
        ao_tests = self.model.get_assignments(pt_tests)
        
        al_tests = self.model2.get_assignments(pt_tests_dupl)
        # a verifier comment cela se comporte ...
        total = len(a_tests)
        nok = 0
        totalg = 0
        okg = 0
        for alt in a_tests:
            if ao_tests(alt.id) != al_tests(alt.id):
                nok += 1
            if ao_tests(alt.id) == "cat1":
                totalg +=1
                if al_tests(alt.id) == "cat1":
                    okg += 1
            # mesure the effect of enforcing constraints in the presence of duplicated criteria
            #import pdb; pdb.set_trace()
            if self.model_with_right_nb_crit == 0:
                for i,j in self.model2.criteria.items():
                    if i[-1] == "d":
                        if not i in self.nb_under_lim_prof_test[self.num_model].keys():
                            self.nb_under_lim_prof_test[self.num_model][i] = 0
                        if self.model2.bpt['b1'].performances[i] > pt_tests_dupl[alt.id].performances[i]:
                            self.nb_under_lim_prof_test[self.num_model][i] += 1
        #import pdb; pdb.set_trace()
        if self.model_with_right_nb_crit == 0:
            for i in self.nb_under_lim_prof_test[self.num_model].keys():
                #import pdb; pdb.set_trace()
                self.nb_under_lim_prof_test[self.num_model][i] = min(self.nb_tests - self.nb_under_lim_prof_test[self.num_model][i],self.nb_under_lim_prof_test[self.num_model][i])
        totalg = 1 if totalg == 0 else totalg
        self.ca_tests_avg[self.num_model] = float(total-nok)/total
        self.ca_good_tests_avg += (float(okg)/totalg)

        return ao_tests,al_tests,(float(total-nok)/total),(float(okg)/totalg)



    def run_mrsort_all_models(self):
        self.report_stats_parameters_csv()
        #lcriteria = []
        classif_tolerance_prop = 0.1
        #print(self.dupl_model_criteria)
        #import pdb; pdb.set_trace()
        for m in range(self.nb_models):
            self.num_model = m
            #Generation de nouvelles alternatives:
            #generation of a new model
            b_inf = (self.nb_alternatives * 1.0 /self.nb_categories)-(classif_tolerance_prop*self.nb_alternatives)
            b_sup = (self.nb_alternatives * 1.0 /self.nb_categories)+(classif_tolerance_prop*self.nb_alternatives)
            notfound = True
            while notfound :
                self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria, fixed_w1 = self.fixed_w1)
                self.dupl_model_criteria = self.model.criteria
                self.pt,self.pt_dupl = generate_random_performance_table_msjp(self.a, self.model.criteria, dupl_crits = self.dupl_model_criteria)
                self.aa = self.model.get_assignments(self.pt)
                i = 1
                size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
                while (size >= b_inf) and (size <= b_sup):
                    if i == self.nb_categories:
                        notfound = False
                        break
                    i += 1
                    size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
            self.pt_dupl_sorted = SortedPerformanceTable(self.pt_dupl)
            self.report_original_model_param_csv()
                        
            self.run_mrsort()
            #test plotting
            #import pdb; pdb.set_trace()
            #self.learned_models.append(self.model2)
            #evaluations
            # how to assign alternatives using model2 (with more criteria than model, how can we compare them)???            
            ca_v,cag_v = self.eval_model_validation()
            #print("model %d : percent. CA = %f" % (m, ca_v))
            #matrix = compute_confusion_matrix(self.aa, self.aa_learned, self.model.categories)
            #print_confusion_matrix(matrix, self.model.categories)

            ao_tests,al_tests,ca_t,cag_t = self.eval_model_test()
            #matrix = compute_confusion_matrix(ao_tests,al_tests,self.model.categories)
            #print_confusion_matrix(matrix,self.model.categories)
            #Statistics
            self.compute_stats_model()
            self.report_stats_model_csv()
        self.report_summary_results_csv()



    def compute_stats_model(self):
        #import pdb; pdb.set_trace()
        self.cpt_dupl_right_crit[self.num_model] = [0,0,0]
        self.cpt_right_crit[self.num_model] = [0,0]
        for i in range(self.nb_criteria): 
            if list(self.model2.criteria.values())[i].dupl_id:
                c_dupl = [j for j in range(len(self.model2.cv.values())) if list(self.model2.criteria.values())[j].id == list(self.model2.criteria.values())[i].dupl_id][0]
                if list(self.model.criteria.values())[i].direction == 1:
                    if float(list(self.model2.cv.values())[c_dupl].value) == 0 and float(list(self.model2.cv.values())[i].value) != 0:
                        self.cpt_dupl_right_crit[self.num_model][0] += 1
                        self.w_dupl_right_crit[self.num_model] += float(list(self.model2.cv.values())[i].value)
                    elif float(list(self.model2.cv.values())[i].value) == 0 and float(list(self.model2.cv.values())[c_dupl].value) == 0:
                        self.cpt_dupl_right_crit[self.num_model][1] += 1
                    else :
                        self.cpt_dupl_right_crit[self.num_model][2] += 1
                    if float(list(self.model2.cv.values())[c_dupl].value) < float(list(self.model2.cv.values())[i].value):
                        self.cpt_gt_right_w[self.num_model] += 1
                # if model.criteria.values()[i].direction == -1: 
                #     if float(model2.cv.values()[i].value) == 0 and float(model2.cv.values()[c_dupl].value) != 0:
                #         nb_right_crit_dir[num_model] += 1
                #         w_right_crit_dir[num_model] += float(model2.cv.values()[c_dupl].value)
            else :
                if list(self.model.criteria.values())[i].direction == 1 and list(self.model2.criteria.values())[i].direction == 1: 
                    if float(list(self.model2.cv.values())[i].value) != 0: 
                        self.cpt_right_crit[self.num_model][0] += 1
                        self.w_right_crit[self.num_model] += float(list(self.model2.cv.values())[i].value)
                    else :
                        self.cpt_right_crit[self.num_model][1] += 1
                # if model.criteria.values()[i].direction == -1 and model2.criteria.values()[i].direction == -1: # does not count since by defaut the direction is 1 and cannot be -1
                #     if float(model2.cv.values()[i].value) != 0:
                #         nb_right_crit_dir[num_model] += 1
                #         w_right_crit_dir[num_model] += float(model2.cv.values()[i].value)
                # if float(model2.cv.values()[i].value) == 0:
                #     nb_null_weights[num_model] += 1



    def report_stats_parameters_csv(self):
        str_noise = "_err" + str(self.noise) if self.noise != None else ""
        str_pretreatment = "_pretreatment" if self.pretreatment else ""
        if self.fixed_w1:
            if self.dir_criteria is None:
                self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_categories))  + "_1w" + str(self.fixed_w1) + str_noise + str_pretreatment + "/"
            else:
                self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.dir_criteria.count(1))) +  "_1w" + str(self.fixed_w1) + str_noise + str_pretreatment + "/"
        else:
            if self.dir_criteria is None:
                self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_categories))  + "_dupl" + str(len(self.l_dupl_criteria)) + str_noise + str_pretreatment + "/"
            else:
                self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.dir_criteria.count(1))) + "-" + str(int(self.dir_criteria.count(-1))) + "_dupl" + str(len(self.l_dupl_criteria)) + str_noise + str_pretreatment + "/"

        if not os.path.exists(DATADIR + self.output_dir):
            os.mkdir(DATADIR + self.output_dir)

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename_valid = "%s/valid_test_dupl_meta_mrsort3-rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_dupl_criteria), dt)
        
        with open(self.filename_valid, "a") as tmp_newfile:
            self.writer = csv.writer(tmp_newfile, delimiter=" ")
            #f = open(filename, 'w')
            #self.writer = csv.writer(f, delimiter = " ")

            self.writer.writerow(['PARAMETERS'])

            #self.writer.writerow([',RANDOMMODE,', RANDOMMODE])
            self.writer.writerow([',nb_alternatives,', self.nb_alternatives])
            self.writer.writerow([',nb_categories,', self.nb_categories])
            self.writer.writerow([',nb_criteria,', self.nb_criteria])

            self.writer.writerow([',nb_outer_loops_meta,', self.meta_l])
            self.writer.writerow([',nb_inner_loops_meta,', self.meta_ll])
            self.writer.writerow([',nb_models_meta,', self.meta_nb_models])
            self.writer.writerow([',nb_learning_models,', self.nb_models])
            self.writer.writerow([',nb_alternatives_test,', self.nb_tests])

            self.writer.writerow([',num_meta_version,', "8"])
            self.writer.writerow([',renewal_method,', self.renewal_method])
            self.writer.writerow([',renewal_models,', self.renewal_models])
            self.writer.writerow([',strategy,', self.strategy])
            self.writer.writerow([',stopping_condition,', str(self.stopping_condition)])
            self.writer.writerow([',decision_rule,', self.decision_rule])

            self.writer.writerow([',gamma,', self.gamma])
            self.writer.writerow([',mu,', self.mu])
            self.writer.writerow([',model_with_right_nb_crit,',self.model_with_right_nb_crit])
            self.writer.writerow([',model_heuristic,',self.model_heuristic])
            self.writer.writerow([',duplication,',self.duplication])
            if self.fixed_w1:
                self.writer.writerow([',pretreatment_fixed_w1,', self.fixed_w1])
            #tmp_newfile.close()


    def report_original_model_param_csv(self):
        with open(self.filename_valid, "a") as tmp_newfile:
            self.writer = csv.writer(tmp_newfile, delimiter=" ")

            self.writer.writerow(['ORIGINAL MODEL'])
            #import pdb; pdb.set_trace()
            self.writer.writerow([',criteria,', ",".join([str(i) for i,j in self.model.criteria.items()])])
            self.writer.writerow([',criteria_direction,', ",".join([str(i.direction) for i in self.model.criteria.values()])])
            self.writer.writerow([',profiles_values,', ",".join([str(self.model.bpt['b1'].performances[i]) for i,j in self.model.criteria.items()])])
            self.writer.writerow([',criteria_weights,', ",".join([str(i.value) for i in self.model.cv.values()])])
            self.writer.writerow([',original_lambda,',self.model.lbda])

            self.writer.writerow(['LEARNING SET'])
            self.writer.writerow([',assignments_id,', ",".join([str(i.id) for i in self.aa])])
            self.writer.writerow([',assignments_cat,',",".join([str(i.category_id) for i in self.aa])])
            self.writer.writerow([',nb_cat1',","+str([str(i.category_id) for i in self.aa].count("cat1"))])
            self.writer.writerow([',nb_cat2',","+str([str(i.category_id) for i in self.aa].count("cat2"))])
            self.writer.writerow([',nb_dupl_criteria,', str(len(self.l_dupl_criteria))])
            if self.noise != None:
                self.writer.writerow([',learning_set_noise,', str(self.noise)])
            #tmp_newfile.close()



    def report_stats_model_csv(self):
        with open(self.filename_valid, "a") as tmp_newfile:
            self.writer = csv.writer(tmp_newfile, delimiter=" ")

            self.writer.writerow(['num_model', self.num_model])

            self.writer.writerow([',criteria,', ",".join([i for i,j in self.model2.criteria.items()])])
            self.writer.writerow([',profiles_values,', ",".join([str(self.model2.bpt['b1'].performances[i]) for i,j in self.model2.criteria.items()])])
            self.writer.writerow([',criteria_weights,', ",".join([str(i.value) for i in self.model2.cv.values()])])
            self.writer.writerow([',lambda,', self.model2.lbda])
            self.writer.writerow([',execution_time,', str(self.exec_time[self.num_model])])

            self.writer.writerow([',nb_dupl_right_crit,', str((self.cpt_dupl_right_crit[self.num_model][0]))])
            self.writer.writerow([',nb_dupl_null_crit,', str((self.cpt_dupl_right_crit[self.num_model][1]))])
            self.writer.writerow([',nb_dupl_GOOD_crit,', str((self.cpt_dupl_right_crit[self.num_model][1]+self.cpt_dupl_right_crit[self.num_model][0]))])
            self.writer.writerow([',nb_dupl_wrong_crit,', str(self.cpt_dupl_right_crit[self.num_model][2])])
            self.writer.writerow([',w_dupl_right_weights,', str(self.w_dupl_right_crit[self.num_model])])
            self.writer.writerow([',nb_right_crit,', str((self.cpt_right_crit[self.num_model][0]))])
            self.writer.writerow([',nb_null_crit,', str((self.cpt_right_crit[self.num_model][1]))])
            self.writer.writerow([',nb_w_right_greater_dupl,', str(float(self.cpt_gt_right_w[self.num_model]))])
            self.writer.writerow([',nb_right_weights,', str(self.w_right_crit[self.num_model])])
            self.writer.writerow([',nb_dir_under_prof,', str(sum(self.nb_under_lim_prof_val[self.num_model].values()))])
            if self.aa_learned:
                self.writer.writerow([',assignments_id,', ",".join([str(i.id) for i in self.aa_learned])])
                self.writer.writerow([',assignments_cat,',",".join([str(i.category_id) for i in self.aa_learned])])

            self.writer.writerow(['LEARNED MODEL (validation)'])
            self.writer.writerow([',CA,', str(self.ca_avg[self.num_model])])
            if self.nb_dupl_criteria:
                if self.model_with_right_nb_crit > 0 or self.model_heuristic:
                    self.writer.writerow([',%_dir_restoration,', str((sum(self.nb_under_lim_prof_val[self.num_model].values()))/self.nb_dupl_criteria)])
                else:
                    self.writer.writerow([',%_dir_restoration,', str((self.cpt_dupl_right_crit[self.num_model][0] + sum(self.nb_under_lim_prof_val[self.num_model].values()))/self.nb_dupl_criteria)])
            #self.writer.writerow([',CA_good,', str(cag_v)])

            self.writer.writerow(['MODEL TEST'])
            self.writer.writerow([',CA_tests,', str(self.ca_tests_avg[self.num_model])])
    #        if self.nb_dupl_criteria:
    #            self.writer.writerow([',%_alt_under_prof_test,', str(sum(self.nb_under_lim_prof_test[self.num_model].values())/self.nb_tests/self.nb_dupl_criteria)])

            #self.writer.writerow([',CA_good_tests,', str(cag_t)])
            #import pdb; pdb.set_trace()
            #tmp_newfile.close()


    def report_summary_results_csv(self):
        with open(self.filename_valid, "a") as tmp_newfile:
            self.writer = csv.writer(tmp_newfile, delimiter=" ")

            self.writer.writerow(['SUMMARY'])
            #writer.writerow(['%_right_crit_dir_avg,', str(float(sum(nb_right_crit_dir))/len(nb_right_crit_dir)/len(model.criteria))])
            #writer.writerow(['w_right_crit_dir_avg,', str(float(sum(w_right_crit_dir))/len(w_right_crit_dir))])
            #writer.writerow(['%_null_weights_avg,', str(float(sum(nb_null_weights))/len(nb_null_weights)/len(model.criteria))])
            self.stats_time = float(sum(self.exec_time))/len(self.exec_time)
            self.writer.writerow([',comput_model_exec_time,' , str(float(sum(self.exec_time))/len(self.exec_time))])
            if self.l_dupl_criteria and self.cpt_dupl_right_crit:
                self.writer.writerow([',%_dupl_right_crit_avg,', str(float(sum([i[1][0] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))]) # prob whenever we encounter a right criteria
                self.writer.writerow([',%_dupl_null_crit_avg,', str(float(sum([i[1][1] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))]) # prob whenever we encounter a null criteria
                self.writer.writerow([',%n_dupl_GODD_crit_avg,', str(float(sum([i[1][1]+i[1][0] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))]) # prob whenever we encounter a right criteria +null criteria = GOOD
            #import pdb; pdb.set_trace()
            self.writer.writerow([',%_dupl_GODD_crit_details_avg,', str([(j[0],round(float(j[1])/self.nb_models,3)) for j in dict(Counter([str(i[1][1]+i[1][0])+"cr" for i in self.cpt_dupl_right_crit.items()])).items()])]) # prob on the group of dupl criteria 

            if self.l_dupl_criteria and self.cpt_dupl_right_crit and len(self.cpt_right_crit) != 0 and self.w_dupl_right_crit:
                self.writer.writerow([',nb_dupl_wrong_crit_avg,', str(float(sum([i[1][2] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))])
                self.writer.writerow([',w_dupl_right_weights_avg,', str(float(sum(self.w_dupl_right_crit))/len(self.w_dupl_right_crit))])
                self.writer.writerow([',nb_right_crit_avg,', str(float(sum([i[1][0] for i in self.cpt_right_crit.items()]))/len(self.cpt_right_crit))])
                self.writer.writerow([',nb_null_crit_avg,', str(float(sum([i[1][1] for i in self.cpt_right_crit.items()]))/len(self.cpt_right_crit))])
            if self.nb_dupl_criteria and self.cpt_gt_right_w and self.w_right_crit:
                self.writer.writerow([',%_w_right_greater_dupl,', str(float(sum(self.cpt_gt_right_w))/self.nb_models/self.nb_dupl_criteria)])
                self.writer.writerow([',w_right_weights_avg,', str(float(sum(self.w_right_crit))/len(self.w_right_crit))])
                self.writer.writerow([',CA_avg,', str(sum(self.ca_avg)/self.nb_models)])
                self.stats_cav = sum(self.ca_avg)/self.nb_models
                #print(self.ca_avg,sum(self.ca_avg),self.nb_models,self.stats_cav)

            self.stats_cag = sum(self.ca_tests_avg)/self.nb_models
            #print(sum(self.ca_tests_avg))
            self.writer.writerow([',CA_tests_avg,', str(sum(self.ca_tests_avg)/self.nb_models)])
            #import pdb; pdb.set_trace()
            if self.model_with_right_nb_crit > 0 or self.model_heuristic:
                if self.nb_dupl_criteria:
                    self.writer.writerow([',%_dir_restoration_avg,', str(float(sum([(sum(self.nb_under_lim_prof_val[i].values())) for i in range(self.nb_models)]))/self.nb_models/self.nb_dupl_criteria)])
            else:
                if self.nb_dupl_criteria and self.cpt_dupl_right_crit:
                    self.stats_capd = float(sum([(sum(self.nb_under_lim_prof_val[i].values()) + self.cpt_dupl_right_crit[i][0]) for i in range(self.nb_models)]))/self.nb_models/self.nb_dupl_criteria
                    self.writer.writerow([',%_dir_restoration_avg,', str(float(sum([(sum(self.nb_under_lim_prof_val[i].values()) + self.cpt_dupl_right_crit[i][0]) for i in range(self.nb_models)]))/self.nb_models/self.nb_dupl_criteria)])
                    if self.nb_dupl_criteria > 1:
                        self.writer.writerow([',%_dir_restoration_avg2,', str(float(sum([1 if sum(self.nb_under_lim_prof_val[i].values()) == self.nb_dupl_criteria else 0 for i in range(self.nb_models)]))/self.nb_models)])
                        self.stats_capd = float(sum([1 if sum(self.nb_under_lim_prof_val[i].values()) == self.nb_dupl_criteria else 0 for i in range(self.nb_models)]))/self.nb_models

                elif not self.cpt_dupl_right_crit:
                    self.writer.writerow([',%_dir_restoration_avg,', str(float(sum([(sum(self.nb_under_lim_prof_val[i].values())) for i in range(self.nb_models)]))/self.nb_models/self.nb_dupl_criteria)])

            #self.writer.writerow([',CA_good_avg,', str(self.ca_good_avg/self.nb_models)])
            if self.nb_dupl_criteria and self.cpt_gt_right_w and self.w_right_crit:
                self.writer.writerow([',CA_avg_std,', str(np.std(self.ca_avg))])
            self.writer.writerow([',CA_tests_avg_std,', str(np.std(self.ca_tests_avg))])
            if self.l_dupl_criteria and self.cpt_dupl_right_crit:
                self.writer.writerow([',%_dir_restoration_std,', str(np.std([(sum(self.nb_under_lim_prof_val[i].values())  + self.cpt_dupl_right_crit[i][0])/len(self.l_dupl_criteria) for i in range(self.nb_models)]))])
    #        if self.nb_dupl_criteria:
    #            self.writer.writerow([',%_alt_under_prof_val_test,', str(sum([sum(i.values()) for i in self.nb_under_lim_prof_test])/self.nb_models/self.nb_tests/self.nb_dupl_criteria)])
            #self.writer.writerow([',CA_good_tests_avg,', str(self.ca_good_tests_avg/self.nb_models)])
            #tmp_newfile.close()



    def report_plot_results_csv(self):

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/plot_results_meta_mrsort3-rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_dupl_criteria), dt)
        f = open(filename, 'w')
        writer = csv.writer(f, delimiter = " ")

        writer.writerow(['SUMMARY DETAILS LIST'])
        writer.writerow([',comput_model_exec_time_list,' , [round(i,2) for i in self.exec_time]])
        writer.writerow([',CA_avg_list,', [round(i,2) for i in self.ca_avg]])
        writer.writerow([',CA_tests_avg_list,', [round(i,2) for i in self.ca_tests_avg]])
        if self.model_with_right_nb_crit > 0 or self.model_heuristic:
            if self.l_dupl_criteria:
                writer.writerow([',CA_dir_restoration,', [round((sum(self.nb_under_lim_prof_val[i].values()))/len(self.l_dupl_criteria),2) for i in range(self.nb_models)]])
            else:
                writer.writerow([',CA_dir_restoration,', [0]*self.nb_models])
        else:
            if self.l_dupl_criteria and self.cpt_dupl_right_crit:
                writer.writerow([',CA_dir_restoration,', [round((sum(self.nb_under_lim_prof_val[i].values()) + self.cpt_dupl_right_crit[i][0])/len(self.l_dupl_criteria),2) for i in range(self.nb_models)]])
                #print(sum([round((sum(self.nb_under_lim_prof_val[i].values()) + self.cpt_dupl_right_crit[i][0])/len(self.l_dupl_criteria),2) for i in range(self.nb_models)])/self.nb_models)
                #writer.writerow([',n_dupl_GODD_crit_avg_list,', [round(sum(i.values())/self.nb_dupl_criteria,2) for i in self.nb_under_lim_prof_val]])
            elif not self.cpt_dupl_right_crit:
                writer.writerow([',CA_dir_restoration,', [round((sum(self.nb_under_lim_prof_val[i].values()))/len(self.l_dupl_criteria),2) for i in range(self.nb_models)]])
            else:
                writer.writerow([',CA_dir_restoration,', [0]*self.nb_models])
        #print(sum(self.nb_heur_inc_positions)/len(self.nb_heur_inc_positions)/self.nb_alternatives)
        #print(sum(self.nb_heur_dec_positions)/len(self.nb_heur_dec_positions)/self.nb_alternatives)
        #print([x for x in self.nb_heur_dec_positions if x is not None]+[x for x in self.nb_heur_inc_positions if x is not None])
        #print([0 for x in self.nb_heur_dec_positions if x is not None]+[1 for x in self.nb_heur_inc_positions if x is not None])
        #print(self.nb_heur_inc_positions)
        #print([0]*len(self.nb_heur_dec_positions) + [1]*len(self.nb_heur_inc_positions))
        #print(self.nb_under_lim_prof_val)
        #print(sum(self.nb_heur_dec_positions), round(sum(self.nb_heur_dec_positions)/self.nb_models,2))
#        x = list(range(self.meta_l))
#        y = [0] * self.meta_l
#        yca = [0] * self.meta_l
#        yca0 = [0] * self.meta_l
#        yca1 = [0] * self.meta_l
#        yca3 = [0] * self.meta_l
#        yca5 = [0] * self.meta_l
#        yca7 = [0] * self.meta_l
#        yca9 = [0] * self.meta_l
#        ypdca = [0] * self.meta_l
#        ypdca0 = [0] * self.meta_l
#        ypdca1 = [0] * self.meta_l
#        ypdca3 = [0] * self.meta_l
#        ypdca5 = [0] * self.meta_l
#        ypdca7 = [0] * self.meta_l
#        ypdca9 = [0] * self.meta_l
#        ca_pop_avg = [0] * self.meta_l
#        ca_pop_avg_best = [0] * self.meta_l
#        import matplotlib.pyplot as plt
#        fig, ax = plt.subplots()
#        
#        #np.mean([i[0] for i in self.proportion_good_models if 0 in i.keys()])
#        for m in range(self.meta_l):
#            #import pdb; pdb.set_trace()
#            #print(m,[i[m] for i in self.proportion_good_models if  m in i.keys()])
#            #y[m] = np.mean([i[m] for i in self.proportion_good_models if m in i.keys()])
#            #ca_prefdir_iterations
#            #y[m] = np.mean([i[m] for i in self.proportion_good_models if len(i.keys()) == 20])
#            y[m] = np.mean([i[m] for i in self.proportion_good_models if m in i.keys()])
#            #print(self.ca_iterations)
#            yca[m] = np.mean([i[m] for i in self.ca_iterations if m in i.keys()])
#            yca0[m] = np.mean([i[m] for i in self.ca_iterations0 if m in i.keys()])
#            yca1[m] = np.mean([i[m] for i in self.ca_iterations1 if m in i.keys()])
#            yca3[m] = np.mean([i[m] for i in self.ca_iterations3 if m in i.keys()])
#            yca5[m] = np.mean([i[m] for i in self.ca_iterations5 if m in i.keys()])
#            yca7[m] = np.mean([i[m] for i in self.ca_iterations7 if m in i.keys()])
#            yca9[m] = np.mean([i[m] for i in self.ca_iterations9 if m in i.keys()])
#            ypdca[m] = np.mean([i[m] for i in self.ca_prefdir_iterations if m in i.keys()])
#            ypdca0[m] = np.mean([i[m] for i in self.ca_prefdir_iterations0 if m in i.keys()])
#            ypdca1[m] = np.mean([i[m] for i in self.ca_prefdir_iterations1 if m in i.keys()])
#            ypdca3[m] = np.mean([i[m] for i in self.ca_prefdir_iterations3 if m in i.keys()])
#            ypdca5[m] = np.mean([i[m] for i in self.ca_prefdir_iterations5 if m in i.keys()])
#            ypdca7[m] = np.mean([i[m] for i in self.ca_prefdir_iterations7 if m in i.keys()])
#            ypdca9[m] = np.mean([i[m] for i in self.ca_prefdir_iterations9 if m in i.keys()])
#            ca_pop_avg[m] = np.mean([i[m] for i in self.ca_pop_avg if m in i.keys()])
#            ca_pop_avg_best[m] = np.mean([i[m] for i in self.ca_pop_avg_best if m in i.keys()])
#            
#            #print([i[m] for i in self.proportion_good_models if len(i.keys()) == 10])
#        #print(y)
#        #print(yca)
#        #print(ypdca)
#        #print(self.end_iterations_meta)
#        #import pdb; pdb.set_trace()
#        
#        #obligatoirement fixer les bornes des axes
#        ax.plot(x,y)
#        print(y)
#        ax.set_xlabel("iteration number")
#        ax.set_ylabel("Avg. nb blue pts")
#        plt.yticks(range(5,10))
#        ax.set_title("Avg. blue pts related to iteration number")
#        plt.xticks(range(0,31,5))
#        directory = DATADIR + self.output_dir
#        #plt.yticks(range(5,11))
#        #plt.xticks(range(0,self.meta_l+1,5))
#        plt.savefig(directory+"/proportion_good_models.png")
#        plt.close()
#        
#        fig, ax = plt.subplots()
#        lines = []
#        lines += ax.plot(x,yca0)
#        lines += ax.plot(x,ca_pop_avg)
#        lines += ax.plot(x,ca_pop_avg_best)
##        lines += ax.plot(x,yca1)
##        lines += ax.plot(x,yca3)
##        lines += ax.plot(x,yca5)
##        lines += ax.plot(x,yca7)
##        lines += ax.plot(x,yca9)
#        print(yca0)
#        print("ca_avg",ca_pop_avg)
#        print("ca_avg_best",ca_pop_avg_best)
#        #ax.legend(lines,["baseline","dr=1mod","dr=3mod","dr=5mod","dr=7mod", "dr=9mod"])
#        ax.legend(lines,["baseline","ca_avg","ca_avg_best"])
#        ax.set_xlabel("iteration number")
#        ax.set_ylabel("Avg. CA restoration")
#        ax.set_title("Avg. CA related to iteration number")
#        plt.xticks(range(0,31,5))
#        plt.yticks(np.arange(0.95,1,0.005))
#        #plt.yticks(np.arange(0.94,1,0.01))
#        #plt.xticks(range(0,self.meta_nb_models,5))
#        plt.savefig(directory+"/comparison_V32_30it_ca.png")
#        plt.close()
#        #import pdb; pdb.set_trace()
#        
#        fig, ax = plt.subplots()
#        lines = []
#        lines += ax.plot(x,ypdca0)
#        lines += ax.plot(x,ypdca1)
#        lines += ax.plot(x,ypdca3)
#        lines += ax.plot(x,ypdca5)
#        lines += ax.plot(x,ypdca7)
#        lines += ax.plot(x,ypdca9)
#        print(ypdca1)
#        #print([np.mean(ypdca0),np.mean(ypdca1),np.mean(ypdca3),np.mean(ypdca5),np.mean(ypdca7),np.mean(ypdca9)])
#        #print([ypdca0[-1],ypdca1[-1],ypdca3[-1],ypdca5[-1],ypdca7[-1],ypdca9[-1]])
#        ax.legend(lines,["baseline","dr=1mod","dr=3mod","dr=5mod","dr=7mod", "dr=9mod"])
#        ax.set_xlabel("iteration number")
#        ax.set_ylabel("Avg. pref dir CA")
#        ax.set_title("Avg. pref dir CA related to iteration number")
#        plt.yticks(np.arange(0.65,0.95,0.05))
#        #plt.yticks(np.arange(0.7,1.01,0.05))
#        #plt.xticksyrange(0,self.meta_nb_models,5))
#        plt.savefig(directory+"/comparison_V32_30it_pdca.png")
#        plt.close()
        
#        fig, ax = plt.subplots()
#        ax.bar(range(11),[i/self.nb_models for i in self.proportion_red_pts_end])
#        print([i/self.nb_models for i in self.proportion_red_pts_end])
#        ax.set_xlabel("models with red pts")
#        ax.set_ylabel("Distribution")
#        ax.set_title("Distribution of nb red pts at the final iteration")
#        #directory = DATADIR + self.output_dir
#        plt.xticks(range(0,11))
#        plt.yticks(np.arange(0,0.8,0.1))
#        plt.savefig(directory+"/proportion_red_pts_end.png")
#        plt.close()
#        
#        fig, ax = plt.subplots()
#        print([i/self.nb_alternatives for i in self.end_iterations_meta])
#        ax.plot(range(30),[(1-i/self.nb_alternatives) for i in self.end_iterations_meta])
#        ax.set_xlabel("nb iterations")
#        ax.set_ylabel("% metaheuristic process ending")
#        ax.set_title("% metaheuristic process ending throughout iterations")
#        #directory = DATADIR + self.output_dir
#        plt.xticks(range(0,31,5))
#        plt.yticks(np.arange(0,1,0.1))
#        plt.savefig(directory+"/proportion_non_opt_models.png")
#        plt.close()
#        #import pdb; pdb.set_trace()


    # build the instance transformed (with duplication of elements of performance table.)
    def build_osomcda_instance_random(self):
        criteria = [f.id for f in self.dupl_model_criteria]
        #import pdb; pdb.set_trace()
        # nb_criteria = len(criteria)*2 if not self.l_dupl_criteria else (self.nb_dupl_criteria
        nb_crits = len(criteria)
        
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/osomcda_rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_dupl_criteria), dt)
                                
        with open(filename, "w") as tmp_newfile:
            out = csv.writer(tmp_newfile, delimiter=" ")
            
            out.writerow(["criterion,direction," + ("," * (nb_crits - 2))])

            for crit in self.model.criteria:
                out.writerow([crit.id + "," + str(1) + "," + ("," * (nb_crits - 2))])
            #for i in range(self.nb_criteria):
            #    if i in self.l_dupl_criteria:
            #        out.writerow([list(self.model.criteria.values())[i].id + "d," + str(-1) + "," + ("," * (nb_crits - 2))])
            out.writerow(["," * (nb_crits)])
            out.writerow(["category,rank," + ("," * (nb_crits - 2))])
            for i in self.model.categories_profiles.to_categories().values():
                out.writerow([i.id + "," + str(i.rank) + "," + ("," * (nb_crits - 2))])
            out.writerow(["," * (nb_crits)])
            out.writerow(["pt," + ",".join(criteria) + ",assignment"])
            for pt_values in self.pt.values():
                #print(pt_values.id)
                #import pdb; pdb.set_trace()
                nrow = [str(pt_values.performances[el.id]) for el in self.model.criteria]
                #dupl_nrow = [str(pt_values.performances[list(self.model.criteria.values())[i].id]) for i in range(self.nb_criteria) if i in self.l_dupl_criteria]
                if self.l_dupl_criteria:
                    out.writerow(["pt" + pt_values.id + "," + ",".join(nrow) + "," + self.aa[pt_values.id].category_id])
                else :
                    out.writerow(["pt" + pt_values.id + "," + ",".join(nrow)  + "," + self.aa[pt_values.id].category_id])
                #tmp_newfile.flush()
        #import pdb; pdb.set_trace()
        return filename


    def learning_process(self):
        self.generate_random_instance()
        self.run_mrsort_all_models()
        self.report_plot_results_csv()
        self.build_osomcda_instance_random()





if __name__ == "__main__":
    DATADIR = os.getenv('DATADIR')
    
    #nb_categories = 2 #fixed
    #nb_criteria = 5
    #nb_alternatives = 100
    #dir_criteria = [1]*nb_criteria # fixed to 1 for all criteria
    #nb_unk_criteria = 1
    #l_dupl_criteria = list(range(nb_criteria))[:nb_unk_criteria]
    
    #nb_tests = 10000
    #nb_models = 100

    #Parameters of the metaheuristic MRSort
    #meta_l = 30
    #meta_ll = 20
    #meta_nb_models = 50
    
    #type of implementation   
    #version_meta = 8
    #renewal_method = 2
    #renewal_models = (0,0.35)
    #strategy = (0.2,25)
    #stopping_condition = meta_l
    #decision_rule = 1


