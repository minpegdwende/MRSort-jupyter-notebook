import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
import datetime
import random
import time
from collections import OrderedDict
import pprint
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import itertools
import copy
from pymcda.electre_tri import MRSort
from fractions import Fraction
import multiprocessing


#MIP MIP FORMULATION : MIP LEARNING FROM SERIES OF RANDOMS MRSORT MODELS
#from itertools import product
from pymcda.learning.mip_mrsort_sp_criteria import MipMRSort

from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import Alternatives, Criteria, Criterion, PerformanceTable
from pymcda.types import AlternativePerformances
from pymcda.types import Criteria, Criterion
from pymcda.types import AlternativeAssignment, AlternativesAssignments, Categories
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
from test_utils import load_mcda_input_data
from pymcda.generate import generate_random_mrsort_model_msjp
from pymcda.generate import generate_random_performance_table_msjp
from pymcda.generate import generate_random_performance_table_msjp_mip
from pymcda.generate import duplicate_performance_table_msjp
from pymcda.generate import generate_categories_profiles
#from pymcda.utils import compute_winning_and_loosing_coalitions
from pymcda.utils import compute_confusion_matrix, print_confusion_matrix, add_errors_in_assignments
#import pdb
from copy import deepcopy
from collections import Counter
from test_utils import test_result, test_results
from test_utils import save_to_xmcda
from pymcda.utils import compute_ca


#DATADIR = os.getenv('DATADIR', '%s/python_workspace/Spyder/oso-pymcda/pymcda-data' % os.path.expanduser('~'))
##DATADIR = os.getenv('DATADIR', '%s/python_workspace/MRSort-jupyter-notebook' % os.path.expanduser('~'))
# RANDOMMODE=0 (1 model generated per unit test +  1 generation of alternatives per unit test), `
# RANDOMMODE=1 (1 model generated per unit test + nb_models of generated alternatives per unit test)
# RANDOMMODE=2 (nb_models models generated per unit test + nb_models of generated alternatives per unit test)
DATADIR = os.getenv('DATADIR')

RANDOMMODE = 2

# meta_mrsort = MetaMRSortVCPop4

class RandMRSortMIPLearning():
    def __init__(self, nb_alternatives, nb_categories, nb_criteria, dir_criteria, l_known_pref_dirs, nb_tests, nb_models, noise = None,version_mip = 2, mip_nb_threads = 1, mip_timetype = 1, mip_timeout = 60):
        self.nb_alternatives = nb_alternatives
        self.nb_categories = nb_categories
        self.nb_criteria = nb_criteria
        self.dir_criteria = dir_criteria
        self.nb_unk_pref_dirs = len(l_known_pref_dirs)
        self.l_known_pref_dirs = l_known_pref_dirs
        self.nb_tests = nb_tests
        self.nb_models = nb_models
        self.version_mip = version_mip
        self.mip_nb_threads = mip_nb_threads
        self.mip_timetype = mip_timetype
        self.mip_timeout = mip_timeout
        self.model = None
        self.pt = None
        self.noise = noise
        self.learned_models = []
        self.ca_avg = [0]*(self.nb_models)
        self.ca_tests_avg = [0]*(self.nb_models)
        self.ca_good_avg = 0
        self.ca_good_tests_avg = 0
        self.ca = 0
        self.cpt_right_crit = dict()
        self.w_right_crit = [0]*(self.nb_models)
        self.w_dupl_right_crit = [0]*(self.nb_models)
        self.nb_null_weights = [0]*(self.nb_models)
        self.exec_time = [0]*(self.nb_models)
        self.cpt_gt_right_w = [0]*(self.nb_models) #count the number of time the good criteria got a greater weight than its duplicated criteria
        self.mip_gamma = [0]*(self.nb_models)
        self.mip_gap = [0]*(self.nb_models)
        self.mip_sumbsp = [0]*(self.nb_models)
        self.mip_obj = [0]*(self.nb_models)
        self.pdca1 = [None]*(self.nb_models)
        self.pdca2 = [None]*(self.nb_models)
        self.mip_sigma = [None]*(self.nb_models)
        self.mip_bm = [None]*(self.nb_models)
        self.mip_b = [None]*(self.nb_models)
        self.cpt_dupl_right_crit = dict()
        self.nb_under_lim_prof_val = [None]*(self.nb_models)
        self.nb_under_lim_prof_test = [None]*(self.nb_models)
        self.nb_heur_inc_positions = [None]*(self.nb_models)
        self.nb_heur_dec_positions = [None]*(self.nb_models)
        self.stats_cav = 0
        self.stats_cag = 0
        self.stats_capd = 0
        self.stats_time = 0
        self.cplex_time =0
        # self stats_pdca1 = 0
        # self stats_pdca2 = 0
        # mettre toutes les autres variable a initialiser a None, car on veut pouvoir tout reinitialiser avec cette function.

    def generate_random_instance(self, classif_tolerance_prop = 0.10):
        b_inf = (self.nb_alternatives * 1.0 /self.nb_categories)-(classif_tolerance_prop*self.nb_alternatives)
        b_sup = (self.nb_alternatives * 1.0 /self.nb_categories)+(classif_tolerance_prop*self.nb_alternatives)
        notfound = False
        modelfound = False
        #ptafound = False
        ptafound = True
        # if self.model :
        #     modelfound = True
        # if self.pt :
        #     ptafound = True
        self.dupl_model_criteria = []
        while notfound :
            if not modelfound:
                #self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria, k=1)
                self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria)
            if not ptafound:
                self.a = generate_alternatives(self.nb_alternatives)
                # if self.duplication :
                #     self.dupl_model_criteria = self.prepare_dupl_criteria_model()
                # else :
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
        #self.pt_dupl_sorted = SortedPerformanceTable(self.pt_dupl)
        # if self.noise != None:
        #     self.aa_noisy = deepcopy(self.aa)
        #     self.aa_err_only = add_errors_in_assignments(self.aa_noisy, self.model.categories, self.noise)
        #self.pt_sorted = SortedPerformanceTable(self.pt)
        

    def run_mrsort(self):
        #categories = self.model.categories_profiles.to_categories()
        #print("enter...deb ")
        #prepare self.model with known and unknown preference directions :
        #for i in l_known_pref_dirs:
        lcriteria = []
        for j in range(self.nb_criteria):
            if j in self.l_known_pref_dirs:
                lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = 0)] #if unknown by default 0 but will be set to 2 just before MIP execution
            else :
                lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = self.dir_criteria[j])]
        #import pdb; pdb.set_trace()
        t1 = time.time()
        #print("mid..deb ")
        if self.noise is None:
            self.model2 = MRSort(Criteria(lcriteria), None, None, None, self.model.categories_profiles, None, None, None)
            mip = MipMRSort(self.model2, self.pt, self.aa, version_mip = self.version_mip, mip_nb_threads = self.mip_nb_threads, mip_timetype = self.mip_timetype, mip_timeout = self.mip_timeout)
            #import pdb; pdb.set_trace()
            #self.model2 = mip.
            #obj,self.model2.bpt2 = mip.solve()
            obj = mip.solve()
            #self.model2_ca = obj / self.nb_alternatives # former manner of computing the CA score : not relevant now ; need to check how obj valued ?
        else:
            self.model2 = MRSort(Criteria(lcriteria), None, None, None, self.model.categories_profiles, None, None, None)
            #import pdb; pdb.set_trace()
            mip = MipMRSort(self.model2, self.pt, self.aa_noisy, version_mip = self.version_mip, mip_nb_threads = self.mip_nb_threads, mip_timetype = self.mip_timetype, mip_timeout = self.mip_timeout)
            #obj,self.model2.bpt2 = mip.solve()
            #print("m1 ")
            obj = mip.solve()
            #print("m2 ")
            #import pdb; pdb.set_trace()
            #print(obj)
            #print("mid..end ")
            self.mip_obj[self.num_model] = obj[0]
            self.mip_gap[self.num_model] = obj[1]
            self.mip_gamma[self.num_model] = obj[2]
            self.mip_sumbsp[self.num_model] = obj[3]
            self.mip_sigma[self.num_model] = obj[5]
            self.mip_bm[self.num_model] = obj[6]
            self.mip_b[self.num_model] = obj[7]
            self.curr_status = obj[8]
            self.cplex_time = obj[9]
            #print(obj[5],obj[6],obj[7])
            
            #setting the preference direction retrieved and true parameters:
            self.pdca1[self.num_model] = dict()
            self.pdca2[self.num_model] = dict()
            for cat_i in  self.model2.bpt.keys():
                for j in range(self.nb_criteria):
                    if j in self.l_known_pref_dirs:
                        #import pdb; pdb.set_trace()
                        sigma = [(i,round(k,5)) for i,k in obj[4] if list(self.model.criteria.values())[j].id==i][0]
                        interval = [round(h,5) for h in self.model2.bpt_sp[cat_i].performances[sigma[0]]]
                        #print(sigma,interval)
                        #import pdb; pdb.set_trace()
                        if sigma[1]==1 and interval[0]==0:
                            self.model2.criteria[sigma[0]].direction = -1
                            self.model2.bpt[cat_i].performances[sigma[0]] = interval[1]
                            # self.pdca1[self.num_model][j] = 1
                            # self.pdca2[self.num_model][j] = 1
                        elif sigma[1]==1 and interval[1]==1:
                            self.model2.criteria[sigma[0]].direction = 1
                            self.model2.bpt[cat_i].performances[sigma[0]] = interval[0]
                            # self.pdca1[self.num_model][j] = 1
                            # self.pdca2[self.num_model][j] = 1
                        # elif sigma[1]==0 and interval[0]==0 and interval[1]!=1:
                        #     self.model2.criteria[sigma[0]].direction = 1
                        #     self.model2.bpt[cat_i].performances[sigma[0]] = interval[1]
                        # elif sigma[1]==0 and interval[1]==1 and interval[0]!=0:
                        #     self.model2.criteria[sigma[0]].direction = -1
                        #     self.model2.bpt[cat_i].performances[sigma[0]] = interval[0]
                        elif sigma[1]==1: #SP
                            self.model2.criteria[sigma[0]].direction = 2
                            # self.pdca1[self.num_model][j] = 1
                            # self.pdca2[self.num_model][j] = 1
                        elif sigma[1]==0: #SV
                            self.model2.criteria[sigma[0]].direction = -2
                            #self.pdca1[self.num_model][j] = -2
                            # self.pdca2[self.num_model][j] = 1
                            # self.model2.criteria[sigma[0]].direction = -2
                            #if self.model2.cv.values())[j][0]==0
                            #print(self.model2.criteria.values())[j]
                        if self.model.criteria[sigma[0]].direction == self.model2.criteria[sigma[0]].direction:
                            self.pdca1[self.num_model][j] = 1
                            self.pdca2[self.num_model][j] = 1
                        else:
                            self.pdca1[self.num_model][j] = 0
                            self.pdca2[self.num_model][j] = 0
                    tmpi = list(self.model.criteria.values())[j].id
                    #import pdb; pdb.set_trace()
                    #self.model2.cv[tmpi].value = round(self.model2.cv[tmpi].value,5)
                    if isinstance(self.model2.bpt_sp[cat_i].performances[tmpi],tuple):
                        self.model2.bpt_sp[cat_i].performances[tmpi] = (round(self.model2.bpt_sp[cat_i].performances[tmpi][0],5),round(self.model2.bpt_sp[cat_i].performances[tmpi][1],5))
                    else:
                        self.model2.bpt_sp[cat_i].performances[tmpi] = round(self.model2.bpt_sp[cat_i].performances[tmpi],5)
            #self.model2.lbda = round(self.model2.lbda,5)
            # print(self.pdca1)
            # print(self.model2.bpt)
            # print(self.model2.criteria)
            # import pdb; pdb.set_trace()
            #print(self.model2.cv)
            #print(self.model2.lbda)
            #self.model2_ca = obj / self.nb_alternatives

        t2 = time.time()
        #self.exec_time[self.num_model] = (t2-t1)
        self.exec_time[self.num_model] = self.cplex_time
        #Retrieving with profiles of SP criteria
        # ap = deepcopy(self.model.bpt["b1"])
        # self.model2.bpt_sp.append(ap)
        
        # for i,j in self.model.criteria.items():
        #     if abs(self.model.criteria[i].direction) == 2:
        #         for cat_i in self.model.bpt.keys():
        #             bb = self.model2.bpt[cat_i].performances[i]
        #             bbsp = self.model2.bpt_sp[cat_i].performances[i]
        #             print("bb",bb, "bbsp",bbsp)
        #             #print("intervalle", bp-binf,bp+binf)
        #             import pdb; pdb.set_trace()
        #             #self.model2.bpt_sp[cat_i].performances[i] = (self.model2.bpt[cat_i].performances[i],(self.model2.b_peak) + self.model2.bpt[cat_i].performances[i])
        #             #self.model2.bpt_sp[cat_i].performances[i] = (bp-binf,bp+binf)
        #             #self.model2.bpt_sp[cat_i].performances[i] = self.model2.bpt2[cat_i].performances[i]
                    
        #print(obj, self.model2_ca)
        #print(self.model2.cv)
        #print(self.model2.bpt_sp)
        #import pdb; pdb.set_trace()
        #print("enter...end ")
        
        return (self.exec_time[self.num_model], self.curr_status)

    
    # assign exmaples with MR-SORT-SP
    def get_assignment_sp(self, tmp_model, ap):
        categories = list(reversed(tmp_model.categories))
        cat = categories[0]
        #import pdb; pdb.set_trace()
        cw = 0
        i=0
        for cat_i in tmp_model.bpt.keys():
            cw = 0
            #for i, profile in enumerate(reversed(tmp_model.profiles)):
            #import pdb; pdb.set_trace()
            for j in tmp_model.cv.keys():
                #import pdb; pdb.set_trace()
                if tmp_model.criteria[j].direction==2:
                    #import pdb; pdb.set_trace()
                    #print(j,cw,ap.performances[j])
                    if ap.performances[j] >= tmp_model.bpt_sp[cat_i].performances[j][0] and ap.performances[j] <= tmp_model.bpt_sp[cat_i].performances[j][1]:
                        cw+=tmp_model.cv[j].value
                elif tmp_model.criteria[j].direction==-2:
                    if ap.performances[j] <= tmp_model.bpt_sp[cat_i].performances[j][0] or ap.performances[j] >= tmp_model.bpt_sp[cat_i].performances[j][1]:
                        cw+=tmp_model.cv[j].value
                else:
                    #print(j,cw,ap.performances[j])
                    if tmp_model.criteria[j].direction==1 and ap.performances[j] >= tmp_model.bpt[cat_i].performances[j]:
                        cw+=tmp_model.cv[j].value
                    if tmp_model.criteria[j].direction==-1 and ap.performances[j] <= tmp_model.bpt[cat_i].performances[j]:
                        cw+=tmp_model.cv[j].value
            # print(ap)
            # print(cw,tmp_model.lbda)
            # print(tmp_model.bpt_sp)
            # print(tmp_model.bpt)
            # import pdb; pdb.set_trace()
            if round(cw,10) >= round(tmp_model.lbda,10):
                break
            cat = categories[i + 1]
            i+=1
        # print(ap)
        # print(cw,tmp_model.lbda)
        # import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        return AlternativeAssignment(ap.id, cat)

    
    def get_assignments_sp(self, tmp_model, tmp_pt):
        aa = AlternativesAssignments()
        for ap in tmp_pt:
            a = self.get_assignment_sp(tmp_model, ap)
            aa.append(a)
        return aa



    def eval_model_validation(self):
        # if 2 in [abs(self.model2.criteria[i].direction) for i in self.model.criteria.keys()]:
        #     self.aa_learned = self.get_assignments_sp(self.model2, self.pt_dupl)
        # else :
        #     self.aa_learned = self.model2.get_assignments(self.pt_dupl)
        self.aa_learned = self.get_assignments_sp(self.model2, self.pt_dupl)
        total = len(self.a)
        nok = 0
        totalg = 0
        okg = 0
        #ref_crit = [i for i,j in self.model2.criteria.items()]
        #tab_max = dict()
        #tab_min = dict()
        #import pdb; pdb.set_trace()
        for alt in self.a:
            if self.aa(alt.id) != self.aa_learned(alt.id):
                nok += 1
                print("misclassified : ",alt.id)
                #import pdb; pdb.set_trace()
                print("details ",self.pt[alt.id])
                #print(self.model2.cv, self.model2.criteria, self.model2.lbda)
            if self.aa(alt.id) == "cat1":
                totalg += 1
                if self.aa_learned(alt.id) == "cat1":
                    okg += 1
            #import pdb; pdb.set_trace()
            #tmpi = 0

        
        #print(self.nb_under_lim_prof_val[self.num_model])       
        #import pdb; pdb.set_trace()
        totalg = 1 if totalg == 0 else totalg
        self.ca_avg[self.num_model] = float(total-nok)/total
        self.ca_good_avg += (float(okg)/totalg)

        return (float(total-nok)/total),(float(okg)/totalg)


    def eval_model_test(self):
        a_tests = generate_alternatives(self.nb_tests)
        #pt_tests = generate_random_performance_table(a_tests, self.dupl_model_criteria)
        pt_tests,pt_tests_dupl = generate_random_performance_table_msjp(a_tests, self.model.criteria, dupl_crits = self.dupl_model_criteria)
        
        #import pdb; pdb.set_trace()
        # if 2 in [abs(self.model2.criteria[i].direction) for i in self.model.criteria.keys()] or 2 in [abs(self.model.criteria[i].direction) for i in self.model.criteria.keys()]:
        #     ao_tests = self.get_assignments_sp(self.model, pt_tests)
        #     al_tests = self.get_assignments_sp(self.model2, pt_tests_dupl)
        # else :
        #     ao_tests = self.model.get_assignments(pt_tests)
        #     al_tests = self.model2.get_assignments(pt_tests_dupl)
        ao_tests = self.get_assignments_sp(self.model, pt_tests)
        al_tests = self.get_assignments_sp(self.model2, pt_tests_dupl)

        #print(self.model2.criteria)
        #import pdb; pdb.set_trace()
        
        
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

        totalg = 1 if totalg == 0 else totalg
        self.ca_tests_avg[self.num_model] = float(total-nok)/total
        self.ca_good_tests_avg += (float(okg)/totalg)

        return ao_tests,al_tests,(float(total-nok)/total),(float(okg)/totalg)


    def random_criteria_weights(self,n,k=3):
        cval = [0]*n
        no_min_w =  True
        while no_min_w:
            random.seed()
            weights = [round(random.random(),k) for i in range(n - 1) ]
            weights += [0,1]
            weights.sort()
            #import pdb; pdb.set_trace()
            if [i for i in range(n) if abs(weights[i]-weights[i+1]) < 0.05] == [] :
                no_min_w = False
            no_min_w = False
        #import pdb; pdb.set_trace()
        for i in range(n):
            cval[i] = round(weights[i+1] - weights[i], k)
        #import pdb; pdb.set_trace()
        return cval
    
    def generate_random_performance_table_msjp_mip_1to1(self, alts, crits, model, k=1, cardinality=10,rep=0.5):
        random.seed()
        pt = PerformanceTable()
        nbcat1= self.nb_alternatives*rep
        nbcat2=self.nb_alternatives-nbcat1
        cpt1=0
        cpt2=0
        for a in alts:
            perfs = {}
            notfound=0
            while notfound<10000:
                random.seed()
                for c in crits:
                    #rdom = round(random.uniform(0, 1), 1)
                    rdom = round(random.choice([f*1./(cardinality) for f in range(0,cardinality+1)]),k)
                    #rdom = round(random.choice([f for f in np.arange(0.05,1,1./(cardinality))]),k)
                    perfs[c.id] = rdom
                ap = AlternativePerformances(a.id, perfs)
                a = self.get_assignment_sp(model, ap)
                #print(a)
                if rep<0:
                    pt.append(ap)
                    break
                if a.category_id=="cat1" and cpt1<nbcat1:
                    pt.append(ap)
                    cpt1+=1
                    break
                if a.category_id=="cat2" and cpt2<nbcat2:
                    pt.append(ap)
                    cpt2+=1
                    break
                notfound+=1
                #print("inf.. "+str(notfound))
            if notfound == 10000:
                break
        return pt,pt


    def run_mrsort_all_models(self, nb_models = 0):
        self.report_stats_parameters_csv()
        lcriteria = []
        if RANDOMMODE == 0:
            self.report_original_model_param_csv()
            if self.model_heuristic:
                for j in range(self.nb_criteria):
                    if j in self.l_dupl_criteria:
                        direction = self.heuristic_preference_directions(list(self.model.criteria.values())[j].id)
                        lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = direction)]
                    else :
                        lcriteria += [Criterion(id=list(self.model.criteria.values())[j].id , direction = 1)]
                self.dupl_model_criteria = Criteria(lcriteria)
        classif_tolerance_prop = 1
        rep=-0.5
        cardinality = 10
        #classif_tolerance_prop = 0
        #print(self.dupl_model_criteria)
        #import pdb; pdb.set_trace()
        for m in range(nb_models):
            #print(m)
            self.num_model = m
            #Generation de nouvelles alternatives:
            #generation of a new model
            self.a = generate_alternatives(self.nb_alternatives)
            if RANDOMMODE != 0:
                b_inf = (self.nb_alternatives * 1.0 /self.nb_categories)-(classif_tolerance_prop*self.nb_alternatives)
                b_sup = (self.nb_alternatives * 1.0 /self.nb_categories)+(classif_tolerance_prop*self.nb_alternatives)
                notfound = True
                # while notfound :
                #     if RANDOMMODE == 2:
                #         self.a = generate_alternatives(self.nb_alternatives)
                #         #self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = [1,1,1,2])
                #         self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria)
                #         #self.a = generate_alternatives(self.nb_alternatives)
                #     #self.dupl_model_criteria = self.prepare_dupl_criteria_model()
                #     #self.pt,self.pt_dupl = generate_random_performance_table_msjp(self.a, self.model.criteria, dupl_crits = self.dupl_model_criteria)
                #     self.pt,self.pt_dupl = generate_random_performance_table_msjp(self.a, self.model.criteria, dupl_crits = [])
                #     #self.aa = self.model.get_assignments(self.pt)
                #     self.aa = self.get_assignments_sp(self.model, self.pt)
                #     i = 1
                #     size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
                #     while (size >= b_inf) and (size <= b_sup):
                #         if i == self.nb_categories:
                #             notfound = False
                #             break
                #         i += 1
                #         size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
                # self.pt_dupl_sorted = SortedPerformanceTable(self.pt_dupl)
            
                #import pdb; pdb.set_trace()
                
                    
                # #import pdb; pdb.set_trace()
                # ### LEARNING SET 1 : Manually generated example  (+,+,{-,+,+-,-+},{-,+,+-,-+}) => 2*2*6*6=144 alternatives 
                # self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria)
                # self.model.lbda = 1.*(self.nb_criteria-1)/(self.nb_criteria)
                # self.model.lbda = round(random.uniform(0.55, 0.95), 1)
                # for i in range(self.nb_criteria):
                #     if i in self.l_known_pref_dirs or abs(self.dir_criteria[i])==2:
                #         self.model.bpt['b1'].performances["c"+str(i+1)] = 0.4
                #         self.model.bpt_sp['b1'].performances["c"+str(i+1)] = (0.4,0.6)
                #         ttmp = tuple(sorted([round(random.uniform(0.05, 0.85), 1),round(random.uniform(0.05, 0.85), 1)]))
                #         self.model.bpt['b1'].performances["c"+str(i+1)] = ttmp[0]
                #         self.model.bpt_sp['b1'].performances["c"+str(i+1)] = (ttmp[0],ttmp[1]+0.1)
                #     else:
                #         self.model.bpt['b1'].performances["c"+str(i+1)] = 0.5
                #         self.model.bpt_sp['b1'].performances["c"+str(i+1)] = 0.5
                #         ttmp = round(random.uniform(0.05, 0.95), 1)
                #         self.model.bpt['b1'].performances["c"+str(i+1)] = ttmp
                #         self.model.bpt_sp['b1'].performances["c"+str(i+1)] = ttmp
                # # self.model.bpt['b1'].performances["c1"] = 0.5
                # # self.model.bpt['b1'].performances["c2"] = 0.5
                # # self.model.bpt['b1'].performances["c3"] = 0.4
                # # self.model.bpt['b1'].performances["c4"] = 0.4
                # # self.model.bpt_sp['b1'].performances["c1"] = 0.5
                # # self.model.bpt_sp['b1'].performances["c2"] = 0.5
                # # self.model.bpt_sp['b1'].performances["c3"] = (0.4,0.6)
                # # self.model.bpt_sp['b1'].performances["c4"] = (0.4,0.6)
                # #import pdb; pdb.set_trace()
                # #self.model.bpt['b1'].performances["c3"] = 0.4 #fictive profile value
                # #self.model.b_peak = 0.5
                # #import pdb; pdb.set_trace()
                # cvals = CriteriaValues()
                # #for i,j in [(1,1./4), (2,1./4), (3,1./4), (4,1./4)]:
                # wcpt = 0
                # for i,j in [(k+1,1./self.nb_criteria) for k in range(self.nb_criteria)]:
                #     cval = CriterionValue()
                #     cval.id = "c"+str(i)
                #     cval.value = j
                #     if i < self.nb_criteria:
                #         if random.random()<0.5 and random.random()<0.5:
                #             cval.value = j-j/4.
                #             wcpt += j/4.
                #         if random.random()>0.5 and random.random()>0.5:
                #             cval.value = j+j/4.
                #             wcpt -= j/4.
                #     else:
                #         cval.value += wcpt
                #     cvals.append(cval)
                # self.model.cv = cvals
                # #import pdb; pdb.set_trace()
                # # pt = PerformanceTable()
                # # i=1
                # # #elems = [[0.05,0.35,0.45,0.55,0.65,0.95] if k in self.l_known_pref_dirs else [0.45,0.55] for k in range(self.nb_criteria)]
                # # elems = [[0.35,0.5,0.65] if k in self.l_known_pref_dirs else [0.45,0.55] for k in range(self.nb_criteria)]
                # # #print(elems)
                # # #import pdb; pdb.set_trace()
                # # for el in itertools.product(*elems):
                # #     #import pdb; pdb.set_trace()
                # #     tmp = {"c"+str(k+1):el[k] for k in range(self.nb_criteria)}
                # #     pt.append(AlternativePerformances('a'+str(i), tmp))
                # #     # if z==0.35:
                # #     #     print(i, 'c1',x, 'c2',y,'c3',z,'c4',a)
                # #     i+=1
                # # pt_dupl = deepcopy(pt)
                # # self.pt = pt
                # # self.pt_dupl = pt_dupl
                # self.pt,self.pt_dupl = generate_random_performance_table_msjp(self.a, self.model.criteria, dupl_crits = [], k=1)
                # self.aa = self.get_assignments_sp(self.model, self.pt)
                # #import pdb; pdb.set_trace()
                # #import pdb; pdb.set_trace()


                #import pdb; pdb.set_trace()
                ### copie of LEARNING SET 1 :except the choice of random preference direction vector made randomly here 
                while notfound :
                    tmp = random.choice([pref_dirs for pref_dirs in itertools.combinations_with_replacement([1,-1,2,-2],self.nb_unk_pref_dirs)]) 
                    #import pdb; pdb.set_trace()
                    if nb_models!=1: # if only one instance, use the current dir_criteria, otherwise draw randomly at each turn
                        self.dir_criteria = list(tmp) + ([1] * (self.nb_criteria-self.nb_unk_pref_dirs))
                    #import pdb; pdb.set_trace()
                    #print(self.dir_criteria)
                    #print(self.dir_criteria)
                    self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria)
                    #self.model.lbda = 1.*(self.nb_criteria-1)/(self.nb_criteria)
                    self.model.lbda = round(random.uniform(0, 1), 2)
                    for i in range(self.nb_criteria):
                        if abs(self.dir_criteria[i])==2:
                            #self.model.bpt['b1'].performances["c"+str(i+1)] = 0.4
                            #self.model.bpt_sp['b1'].performances["c"+str(i+1)] = (0.4,0.6)
                            #ttmp = tuple(sorted([round(random.uniform(0, 0), 1),round(random.uniform(0.05, 0.95), 1)]))
                            if self.dir_criteria[i]==2:
                                ttmp = tuple(sorted([round(random.choice([f*1./(cardinality) for f in range(1,cardinality-1)]),1),round(random.choice([f*1./(cardinality) for f in range(1,cardinality-1)]),1)]))
                            else:
                                ttmp = tuple(sorted([round(random.choice([f*1./(cardinality) for f in range(0,cardinality)]),1),round(random.choice([f*1./(cardinality) for f in range(0,cardinality)]),1)]))
                            #ttmp = tuple(sorted([round(random.choice([f for f in np.arange(0.05,1.,1./(cardinality))]),2), round(random.choice([f for f in np.arange(0.05,1.,1./(cardinality))]),2)]))
                            self.model.bpt['b1'].performances["c"+str(i+1)] = ttmp[0]
                            self.model.bpt_sp['b1'].performances["c"+str(i+1)] = (ttmp[0],ttmp[1]+round(1./(cardinality),1))
                        else:
                            #self.model.bpt['b1'].performances["c"+str(i+1)] = 0.5
                            #self.model.bpt_sp['b1'].performances["c"+str(i+1)] = 0.5
                            #ttmp = round(random.uniform(0.05, 0.95), 1)
                            ttmp = round(random.choice([f*1./(cardinality) for f in range(1,cardinality)]),1)
                            #ttmp = round(random.choice([f for f in np.arange(0.05,1.,1./(cardinality))]),2)
                            self.model.bpt['b1'].performances["c"+str(i+1)] = ttmp
                            self.model.bpt_sp['b1'].performances["c"+str(i+1)] = ttmp
                            #import pdb; pdb.set_trace()
                    # self.model.bpt['b1'].performances["c1"] = 0.5
                    # self.model.bpt['b1'].performances["c2"] = 0.5
                    # self.model.bpt['b1'].performances["c3"] = 0.4
                    # self.model.bpt['b1'].performances["c4"] = 0.4
                    # self.model.bpt_sp['b1'].performances["c1"] = 0.5
                    # self.model.bpt_sp['b1'].performances["c2"] = 0.5
                    # self.model.bpt_sp['b1'].performances["c3"] = (0.4,0.6)
                    # self.model.bpt_sp['b1'].performances["c4"] = (0.4,0.6)
                    #import pdb; pdb.set_trace()
                    #self.model.bpt['b1'].performances["c3"] = 0.4 #fictive profile value
                    #self.model.b_peak = 0.5
                    #import pdb; pdb.set_trace()
                    cvals = CriteriaValues()
                    #for i,j in [(1,1./4), (2,1./4), (3,1./4), (4,1./4)]:
                    tmp = self.random_criteria_weights(self.nb_criteria,k=2)
                    wcpt = 0
                    for i,j in [(k+1,1./self.nb_criteria) for k in range(self.nb_criteria)]:
                        cval = CriterionValue()
                        cval.id = "c"+str(i)
                        #cval.value = j
                        cval.value = tmp[i-1]
                        # if i < self.nb_criteria:
                        #     if random.random()<0.5 and random.random()<0.5:
                        #         cval.value = j-j/4.
                        #         wcpt += j/4.
                        #     if random.random()>0.5 and random.random()>0.5:
                        #         cval.value = j+j/4.
                        #         wcpt -= j/4.
                        # else:
                        #     cval.value += wcpt
                        cvals.append(cval)
                    self.model.cv = cvals
                    #import pdb; pdb.set_trace()
                    # pt = PerformanceTable()
                    # i=1
                    # #elems = [[0.05,0.35,0.45,0.55,0.65,0.95] if k in self.l_known_pref_dirs else [0.45,0.55] for k in range(self.nb_criteria)]
                    # elems = [[0.35,0.5,0.65] if k in self.l_known_pref_dirs else [0.45,0.55] for k in range(self.nb_criteria)]
                    # #print(elems)
                    # #import pdb; pdb.set_trace()
                    # for el in itertools.product(*elems):
                    #     #import pdb; pdb.set_trace()
                    #     tmp = {"c"+str(k+1):el[k] for k in range(self.nb_criteria)}
                    #     pt.append(AlternativePerformances('a'+str(i), tmp))
                    #     # if z==0.35:
                    #     #     print(i, 'c1',x, 'c2',y,'c3',z,'c4',a)
                    #     i+=1
                    # pt_dupl = deepcopy(pt)
                    # self.pt = pt
                    # self.pt_dupl = pt_dupl
                    #self.pt,self.pt_dupl = generate_random_performance_table_msjp(self.a, self.model.criteria, dupl_crits = [], k=1)
                    #self.pt,self.pt_dupl = generate_random_performance_table_msjp_mip(self.a, self.model.criteria, dupl_crits = [], k=2, cardinality=cardinality)
                    self.pt,self.pt_dupl = self.generate_random_performance_table_msjp_mip_1to1(self.a, self.model.criteria,self.model, k=2, cardinality=cardinality,rep=rep) #rep=% of cat1
                    self.aa = self.get_assignments_sp(self.model, self.pt)
                    i = 1
                    size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
                    if size == self.nb_alternatives*rep or rep<0:
                        notfound = False
                    #print(notfound)
                    #import pdb; pdb.set_trace()
                    # while (size >= b_inf) and (size <= b_sup):
                    #     if i == self.nb_categories:
                    #         notfound = False
                    #         break
                    #     i += 1
                    #     size = len(self.aa.get_alternatives_in_category('cat'+str(i)))
                #import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()
                
                
                
                # #import pdb; pdb.set_trace()
                # ## LEARNING SET 4 specific : ASA
                # ##Execution d'une instance fichier : ici ASA et modification des parametres de base.
                # ##useful commmand line for extracting desired criteria
                # ###cut -d , -f 1,2,3,4,19,20 asa_binarized_mip2.csv > asa_binarized_mip2_new.csv 
                # #data = self.load_mcda_input_data_ASA("/Users/pegdwendeminoungou/python_workspace/Spyder/oso-pymcda/datasets/pegdwende_faulty1.csv")
                # #data = self.load_mcda_input_data_ASA("/Users/pegdwendeminoungou/python_workspace/Spyder/oso-pymcda/datasets/asa_binarized_12vs34.csv")
                # #import pdb; pdb.set_trace()
                # data = load_mcda_input_data("/Users/pegdwendeminoungou/python_workspace/Spyder/oso-pymcda/datasets/handmade_mip_sv.csv")
                # #data = load_mcda_input_data("/Users/pegdwendeminoungou/python_workspace/Spyder/oso-pymcda/datasets/asa_binarized_12vs34_mip1_age103_sys24_dia17_gly82_"+str(self.nb_alternatives)+"alts.csv")
                # self.a = data.a
                # self.aa = data.aa
                # #import pdb; pdb.set_trace()
                # #self.c = data.c
                # self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria)
                # self.model.lbda = 1.*(self.nb_criteria-1)/(self.nb_criteria)
                # for i in range(self.nb_criteria):
                #     if i in self.l_known_pref_dirs or abs(self.dir_criteria[i])==2:
                #         self.model.bpt['b1'].performances["c"+str(i+1)] = 0.4
                #         self.model.bpt_sp['b1'].performances["c"+str(i+1)] = (0.4,0.6)
                #     else:
                #         self.model.bpt['b1'].performances["c"+str(i+1)] = 0.5
                #         self.model.bpt_sp['b1'].performances["c"+str(i+1)] = 0.5
                # cvals = CriteriaValues()
                # #for i,j in [(1,1./4), (2,1./4), (3,1./4), (4,1./4)]:
                # for i,j in [(k+1,1./self.nb_criteria) for k in range(self.nb_criteria)]:
                #     cval = CriterionValue()
                #     cval.id = "c"+str(i)
                #     cval.value = j
                #     cvals.append(cval)
                # self.model.cv = cvals
                # #import pdb; pdb.set_trace()
                # pt_dupl = deepcopy(data.pt)
                # self.pt = data.pt 
                # self.pt_dupl = pt_dupl
                # #import pdb; pdb.set_trace()
                # #self.aa = self.get_assignments_sp(self.model, self.pt)
                # # #import pdb; pdb.set_trace()
                        
                        
            self.report_original_model_param_csv()
            if self.noise != None:
                self.aa_noisy = deepcopy(self.aa)
                self.aa_err_only = add_errors_in_assignments(self.aa_noisy, self.model.categories, self.noise)

            res1, status = self.run_mrsort()
            #print(status)
            if status==107:
                #print("here")
                self.mip_obj[self.num_model] = None
                self.mip_gap[self.num_model] = None
                self.mip_gamma[self.num_model] = None
                self.mip_sumbsp[self.num_model] = None
                self.mip_sigma[self.num_model] = None
                self.mip_bm[self.num_model] = None
                self.mip_b[self.num_model] = None
                self.pdca1[self.num_model] = None
                self.pdca2[self.num_model] = None
                self.exec_time[self.num_model] = None
                self.ca_avg[self.num_model] = None
                self.ca_tests_avg[self.num_model] = None
                self.model2 = None
            else:
                #print(stopped," not stopped")
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
                #self.compute_stats_model()
                self.report_stats_model_csv()
        self.report_summary_results_csv()
        
        
    def load_mcda_input_data_ASA(self, filepath):
        #result = filepath.split(".")[0]+"_tmp.csv"
        result = filepath.split(".")[0]+"_____.csv"
        #result = filepath.split(".")[0]+"_mip1_age103_sys24_dia17_gly82_alts_7crits.csv"
        sys = set()
        dia = set()
        gly = set()
        age = set()       
        sys_hist = []
        dia_hist = []
        age_hist = []
        gly_hist = []
        gly_hist0 = []
        gly_hist1 = []
        cpt=[0,0]
        oplp = ""
        opld = ""
        cpt_patient = 1
        with open(filepath,"r") as tmp_file:
            rdr= csv.reader(tmp_file)
            #import pdb; pdb.set_trace()
            with open(result,"w") as tmp_file2:
                wtr= csv.writer(tmp_file2)
                #i=0
                for r in rdr:
                    #wtr.writerow((r[0],r[1],r[2],r[3],r[4],r[5],r[7]))
                    #import pdb; pdb.set_trace()
                    if len(r)>5 and r[6]!="assignment" and r[6]!="":
                        # wtr.writerow((r[0],float(r[1]),float(r[2]),float(r[3]),float(r[4]),1 if r[6]=="cat1" else 2))
                        # else:
                        #wtr.writerow(r)
                        #import pdb; pdb.set_trace()
                        wtr.writerow((r[0],r[1],r[2],r[3],r[4],r[6]))
                        oplp += str([float(r[1]),float(r[2]),float(r[3]),float(r[4])])+", "
                        opld += str(1 if r[6]=="cat1" else 2)+", "
                        #print(r[6])
                        #import pdb; pdb.set_trace()
                    else:
                        wtr.writerow(r)
                    #if i==1 or i==2 or i==3 or i==4 or i==5 or i==14 or i==18:
                    #print(r[10:12])
                    #ASA from this line till the end
                    # if r[23]!="assignment" and r[23]!="":
                    #     #if r[0] != "p234":
                    #     #print(r[0])
                    #     # age.add(round(((float(r[1]))/105.),2)) #age94_sys24_dia17_gly52_898_ASA_12VS34
                    #     # sys.add(round((((float(r[20])-9))/11.5),2)) #age94_sys24_dia17_gly52_898_ASA_12VS34
                    #     # dia.add(round((((float(r[21])-5))/8.),2)) #age94_sys24_dia17_gly52_898_ASA_12VS34
                    #     # gly.add(round((((float(r[17])-0.5))/3.3),2)) #age94_sys24_dia17_gly52_898_ASA_12VS34
                        
                    #     # age.add(round(((float(r[1])//10)/10.),2)) #age11_sys6_dia9_gly28_898_ASA_12VS34
                    #     # sys.add(round((((float(r[20])-9)//2)/10),2)) #age11_sys6_dia9_gly28_898_ASA_12VS34
                    #     # dia.add(round((((float(r[21])-5)//1)/10.),2)) #age11_sys6_dia9_gly28_898_ASA_12VS34
                    #     # gly.add(round((((float(r[17])-0.5)//0.1)/100),2)) #age11_sys6_dia9_gly28_898_ASA_12VS34
                        
                    #     age.add(round(((float(r[1]))/105.),3)) #age103_sys24_dia17_gly82_700_ASA_12VS34_ASA_12VS34
                    #     sys.add(round((((float(r[20])-9))/11.5),3)) #age103_sys24_dia17_gly82_700_ASA_12VS34
                    #     dia.add(round((((float(r[21])-5))/8.),3)) #age103_sys24_dia17_gly82_700_ASA_12VS34
                    #     gly.add(round((((float(r[17])-0.5))/3.3),3)) #age103_sys24_dia17_gly82_700_ASA_12VS34
                    #     # if float(r[0][1:])<=700:
                    #     if True or (float(r[17])>=1. and float(r[17])<1.2 and r[23]=="1") or float(r[17])<1 or float(r[17])>=1.2:
                    #     #if (float(r[17])>=1. and float(r[17])<1.2 and r[23]=="1") or (float(r[17])<1 and r[23]=="0") or float(r[17])>=1.2:
                    #         #if r[23]=="1":
                    #         #if random.random()>0:
                    #         dia_hist += [float(r[21])]
                    #         sys_hist += [float(r[20])]
                    #         age_hist += [float(r[1])]                            
                    #         gly_hist += [float(r[17])]
                    #         if r[23]=="0":
                    #             gly_hist0 += [float(r[17])]
                    #         else:
                    #             gly_hist1 += [float(r[17])]
                    #         #wtr.writerow((r[0], round((float(r[1])//10)/10.,2), r[2], r[3], r[4], r[13], round(((float(r[20])-9)//2)/10,2), round(((float(r[21])-5)//1)/10., 2), round(((float(r[17])-0.5)//0.1)/100,2), "cat2" if r[23]=="1" else "cat1" if r[23]=="0" else "" ))
                    #         wtr.writerow(("p"+str(cpt_patient), round((float(r[1]))/105.,3), r[2], r[3], r[4], r[13], round(((float(r[20])-9))/11.5,3), round(((float(r[21])-5))/8., 3), round(((float(r[17])-0.5))/3.3,3), "cat2" if r[23]=="1" else "cat1" if r[23]=="0" else "" ))
                    #         #wtr.writerow((r[0],r[1],r[2],r[3],r[4],r[5], r[6], r[7], r[8], r[9],r[10], r[11], r[12], r[13], r[14],r[15], r[16], r[17], r[18], r[19],r[20], r[21], r[22], "1" if r[23]=="cat2" else "0" if  r[23]=="cat1" else "" ))
                    #         cpt_patient+=1

                        
                    #     #wtr.writerow((r[0], round((float(r[1])//10)/10.,3), r[2], r[3], r[4], r[13], round((float(r[20])//1)/25,3), round((float(r[21])//1)/25, 3), round((float(r[17])//0.5)/10,3), "cat2" if r[23]=="1" else "cat1" if r[23]=="0" else "" ))
                    #     #wtr.writerow((r[0], round(float(r[1])/125,3), r[2], r[3], r[4], r[13], round(float(r[20])/25,3), round(float(r[21])/25, 3), round(float(r[17])/5,3), "cat2" if r[23]=="1" else "cat1" if r[23]=="0" else "" ))
                    #     #wtr.writerow((r[0], r[1], r[2], r[3], r[4], r[13], r[20], r[21], r[17], (r[11] if r[10]=="70" else r[10]), "cat2" if r[23]=="1" else "cat1" if r[23]=="0" else ""))
                    # else:
                    #     wtr.writerow((r[0], r[1], r[2], r[3], r[4], r[13], r[20], r[21], r[17], r[23]))
                    #     #wtr.writerow(r)
                    #     #print(r[20],r[21])
                    #     #wtr.writerow((r[0], r[1], r[2], r[3], r[4], r[13], r[20], r[21], r[17], (r[11] if r[10]=="70" else r[10]), r[23]))
                    #i+=1
        print("age",len(age),age)
        print("sys",len(sys),sys)
        print("dia",len(dia),dia)
        print("gly",len(gly),gly)
        print(oplp)
        print(opld)
        print(len(set(age_hist)),len(set(dia_hist)),len(set(sys_hist)),len(set(gly_hist)))
        import pdb; pdb.set_trace()
        print(len(gly_hist0),len(gly_hist1))
        nb=len(set(gly_hist))
        #plt.hist(gly_hist, bins=nb)
        fig, ax = plt.subplots()
        plt.hist([gly_hist1, gly_hist0], stacked=True, bins=nb, label=['Accepted','Refused'])
        plt.axvline(np.median(gly_hist), color='k', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(gly_hist,25), color='k', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(gly_hist,75), color='k', linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(gly_hist), color='red', linestyle='dashed', linewidth=1)
        # plt.yscale('log')
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        #ax.set_yticks([2,5,10,50,100,200])
        ax.set_yticks([25,50,75,100,125,150,175,200,225,250])
        ax.set_xticks(np.arange(0.5, 4, 0.5))
        #ticker.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        plt.xlabel('Glycemia')
        plt.ylabel('#values')
        plt.title("Histogramm glycemia, #values:"+str(nb))
        plt.legend()
        plt.savefig("/Users/pegdwendeminoungou/python_workspace/Spyder/oso-pymcda/pymcda-data/histogramm_glycemia_binarized")

        import pdb; pdb.set_trace()


    def report_stats_parameters_csv(self):
        str_noise = "_err" + str(int(self.noise*100)) if self.noise != None else ""
        #str_pretreatment = "_pretreatment" if self.pretreatment else ""
        str_pretreatment = ""
        # if self.fixed_w1:
        #     if self.dir_criteria is None:
        #         self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_categories))  + "_1w" + str(self.fixed_w1) + str_noise + str_pretreatment + "/"
        #     else:
        #         self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.dir_criteria.count(1))) +  "_1w" + str(self.fixed_w1) + str_noise + str_pretreatment + "/"
        # else:
        if self.dir_criteria is None:
            self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_categories))  + "_dupl" + str(len(self.l_dupl_criteria)) + str_noise + str_pretreatment + "/"
        else:
            self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.dir_criteria.count(1))) + "-" + str(int(self.dir_criteria.count(-1))) + "_upd" + str(len(self.l_known_pref_dirs)) + str_noise + str_pretreatment + "/"
            self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_criteria)) + "-" + str(int(self.dir_criteria.count(1))) + "-" + str(int(self.dir_criteria.count(-1))) + "-" + str(int(self.dir_criteria.count(2))) + "-" + str(int(self.dir_criteria.count(-2))) + "_dupl" + str(len(self.l_known_pref_dirs)) + str_noise + str_pretreatment + "/"
        if not os.path.exists(DATADIR + self.output_dir):
            os.mkdir(DATADIR + self.output_dir)

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename_valid = "%s/valid_test_dupl_meta_mrsort3-rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_known_pref_dirs), dt)
        
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
    
            writer.writerow(['PARAMETERS'])
    
            writer.writerow([',RANDOMMODE,', RANDOMMODE])
            writer.writerow([',nb_alternatives,', self.nb_alternatives])
            writer.writerow([',nb_categories,', self.nb_categories])
            writer.writerow([',nb_criteria,', self.nb_criteria])
    
            writer.writerow([',nb_learning_models,', self.nb_models])
            writer.writerow([',nb_alternatives_test,', self.nb_tests])
            
            writer.writerow([',mip_version,', 0])
                
            
    def report_original_model_param_csv(self):
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            
            writer.writerow(['ORIGINAL MODEL'])
            #import pdb; pdb.set_trace()
            writer.writerow([',criteria,', ",".join([str(i) for i,j in self.model.criteria.items()])])
            writer.writerow([',criteria_direction,', ",".join([str(i.direction) for i in self.model.criteria.values()])])
            for cat_i in self.model.bpt.keys():
                writer.writerow([',profiles_values_'+cat_i+',', ",".join([str(self.model.bpt_sp[cat_i].performances[i][0]) if abs(j.direction)==2 else str(self.model.bpt[cat_i].performances[i]) for i,j in self.model.criteria.items()])])
                writer.writerow([',profiles2_values_'+cat_i+',', ",".join([str(self.model.bpt_sp[cat_i].performances[i][1]) if abs(j.direction)==2 else "" for i,j in self.model.criteria.items()])])

            writer.writerow([',criteria_weights,', ",".join([str(i.value) for i in self.model.cv.values()])])
            writer.writerow([',original_lambda,',self.model.lbda])
    
            writer.writerow(['LEARNING SET'])
            writer.writerow([',assignments_id,', ",".join([str(i.id) for i in self.aa])])
            writer.writerow([',assignments_cat,',",".join([str(i.category_id) for i in self.aa])])
            writer.writerow([',nb_cat1',","+str([str(i.category_id) for i in self.aa].count("cat1"))])
            writer.writerow([',nb_cat2',","+str([str(i.category_id) for i in self.aa].count("cat2"))])
            writer.writerow([',nb_unk_pref_dirs,', str(len(self.l_known_pref_dirs))])
            if self.noise != None:
                writer.writerow([',learning_set_noise,', str(self.noise)])
    


    def report_stats_model_csv(self):
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")

            writer.writerow(['LEARNED MODEL num_model', self.num_model])
    
            writer.writerow([',criteria,', ",".join([i for i,j in self.model2.criteria.items()])])
            writer.writerow([',criteria_direction,', ",".join([str(i.direction) for i in self.model2.criteria.values()])])
            for cat_i in self.model.bpt.keys():
                writer.writerow([',profiles_values_'+cat_i+',', ",".join([str(self.model2.bpt_sp[cat_i].performances[i][0]) if abs(j.direction)==2 else str(self.model2.bpt[cat_i].performances[i]) for i,j in self.model2.criteria.items()])])
                writer.writerow([',profiles2_values_'+cat_i+',', ",".join([str(self.model2.bpt_sp[cat_i].performances[i][1]) if abs(j.direction)==2 else "" for i,j in self.model2.criteria.items()])])
                #writer.writerow([',profiles2_values_'+cat_i+',', ",".join([str(self.model2.bpt_sp[cat_i].performances[i][1]) if abs(j.direction)==2 or [m for m,n in self.model2.criteria.items()].index(i) in self.l_known_pref_dirs else "" for i,j in self.model2.criteria.items()])])

            writer.writerow([',criteria_weights,', ",".join([str(i.value) for i in self.model2.cv.values()])])
            writer.writerow([',lambda,', self.model2.lbda])
            writer.writerow([',execution_time,', str(self.exec_time[self.num_model])])
            
            # writer.writerow([',nb_dupl_right_crit,', str((self.cpt_dupl_right_crit[self.num_model][0]))])
            # writer.writerow([',nb_dupl_null_crit,', str((self.cpt_dupl_right_crit[self.num_model][1]))])
            # writer.writerow([',nb_dupl_GOOD_crit,', str((self.cpt_dupl_right_crit[self.num_model][1]+self.cpt_dupl_right_crit[self.num_model][0]))])
            # writer.writerow([',nb_dupl_wrong_crit,', str(self.cpt_dupl_right_crit[self.num_model][2])])
            # writer.writerow([',w_dupl_right_weights,', str(self.w_dupl_right_crit[self.num_model])])
            # writer.writerow([',nb_right_crit,', str((self.cpt_right_crit[self.num_model][0]))])
            # writer.writerow([',nb_null_crit,', str((self.cpt_right_crit[self.num_model][1]))])
            # writer.writerow([',nb_w_right_greater_dupl,', str(float(self.cpt_gt_right_w[self.num_model]))])
            # writer.writerow([',nb_right_weights,', str(self.w_right_crit[self.num_model])])
            # writer.writerow([',nb_dir_under_prof,', str(sum(self.nb_under_lim_prof_val[self.num_model].values()))])
            
            if self.aa_learned:
                #import pdb; pdb.set_trace()
                writer.writerow([',assignments_id,', ",".join([str(i.id) for i in self.aa_learned])])
                writer.writerow([',assignments_cat,',",".join([str(i.category_id) for i in self.aa_learned])])
    
            writer.writerow(['Results validation'])
            writer.writerow([',CA,', str(self.ca_avg[self.num_model])])
            if self.l_known_pref_dirs:
                writer.writerow([',PDCA1,', str((sum(self.pdca1[self.num_model].values()))//len(self.l_known_pref_dirs))])
                writer.writerow([',PDCA2,', str((sum(self.pdca2[self.num_model].values()))/len(self.l_known_pref_dirs))])
            #writer.writerow([',CA_good,', str(cag_v)])
            writer.writerow([',mip_gamma,', str(self.mip_gamma[self.num_model])])
            writer.writerow([',mip_gap,', str(self.mip_gap[self.num_model])])
            writer.writerow([',mip_sumbsp,', str(self.mip_sumbsp[self.num_model])])
            writer.writerow([',mip_sigma,', str(self.mip_sigma[self.num_model])])
            writer.writerow([',mip_bm,', str(self.mip_bm[self.num_model])])
            writer.writerow([',mip_b,', str(self.mip_b[self.num_model])])

            writer.writerow(['Results test'])
            writer.writerow([',CA_tests,', str(self.ca_tests_avg[self.num_model])])


    def report_summary_results_csv(self):
        with open(self.filename_valid, "a") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")

            writer.writerow(['SUMMARY'])
            self.stats_time = self.exec_time
            #import pdb; pdb.set_trace()
            writer.writerow([',exec_time_avg,' , str(np.mean([x for x in self.stats_time if x is not None]))])
            writer.writerow([',exec_time_std,' , str(np.std([x for x in self.stats_time if x is not None]))])

            # if self.l_dupl_criteria and self.cpt_dupl_right_crit:
            #     writer.writerow([',%_dupl_right_crit_avg,', str(float(sum([i[1][0] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))]) # prob whenever we encounter a right criteria
            #     writer.writerow([',%_dupl_null_crit_avg,', str(float(sum([i[1][1] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))]) # prob whenever we encounter a null criteria
            #     writer.writerow([',%n_dupl_GODD_crit_avg,', str(float(sum([i[1][1]+i[1][0] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))]) # prob whenever we encounter a right criteria +null criteria = GOOD
            # writer.writerow([',%_dupl_GODD_crit_details_avg,', str([(j[0],round(float(j[1])/self.nb_models,3)) for j in dict(Counter([str(i[1][1]+i[1][0])+"cr" for i in self.cpt_dupl_right_crit.items()])).items()])]) # prob on the group of dupl criteria 
            # if self.l_dupl_criteria and self.cpt_dupl_right_crit and len(self.cpt_right_crit) != 0 and self.w_dupl_right_crit:
            #     writer.writerow([',nb_dupl_wrong_crit_avg,', str(float(sum([i[1][2] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))])
            #     writer.writerow([',w_dupl_right_weights_avg,', str(float(sum(self.w_dupl_right_crit))/len(self.w_dupl_right_crit))])
            #     writer.writerow([',nb_right_crit_avg,', str(float(sum([i[1][0] for i in self.cpt_right_crit.items()]))/len(self.cpt_right_crit))])
            #     writer.writerow([',nb_null_crit_avg,', str(float(sum([i[1][1] for i in self.cpt_right_crit.items()]))/len(self.cpt_right_crit))])
            # if self.nb_dupl_criteria and self.cpt_gt_right_w and self.w_right_crit:
            #     writer.writerow([',%_w_right_greater_dupl,', str(float(sum(self.cpt_gt_right_w))/self.nb_models/self.nb_dupl_criteria)])
            #     writer.writerow([',w_right_weights_avg,', str(float(sum(self.w_right_crit))/len(self.w_right_crit))])
                
            self.stats_cav = self.ca_avg
            writer.writerow([',CAv_avg,', str(np.mean([x for x in self.ca_avg if x is not None]))])
            writer.writerow([',CAv_std,', str(np.std([x for x in self.ca_avg if x is not None]))])
            
            self.stats_cag = self.ca_tests_avg    
            writer.writerow([',CAg_avg,', str(np.mean([x for x in self.ca_tests_avg if x is not None]))])
            writer.writerow([',CAg_std,', str(np.std([x for x in self.ca_tests_avg if x is not None]))])
            
            writer.writerow([',mip_gamma_avg,', str(np.mean([x for x in self.mip_gamma if x is not None]))])
            writer.writerow([',mip_gap_avg,', str(np.mean([x for x in self.mip_gap if x is not None]))])

    
            #import pdb; pdb.set_trace()
            # self.stats_pdca2 = [sum(self.nb_under_lim_prof_val[i].values())/self.nb_dupl_criteria for i in range(self.nb_models)]
            # if self.nb_dupl_criteria == 1:
            #     self.stats_pdca1 = self.stats_pdca2
            #     writer.writerow([',PDCA1_avg,', str(np.mean(self.stats_pdca1))])
            #     writer.writerow([',PDCA1_std,', str(np.std(self.stats_pdca1))])
            #     writer.writerow([',PDCA2_avg,', str(np.mean(self.stats_pdca2))])
            #     writer.writerow([',PDCA2_std,', str(np.std(self.stats_pdca2))])
            # elif self.nb_dupl_criteria > 1:
            #     self.stats_pdca1 = [sum(self.nb_under_lim_prof_val[i].values())//self.nb_dupl_criteria for i in range(self.nb_models)]
            #     writer.writerow([',PDCA1_avg,', str(np.mean(self.stats_pdca1))])
            #     writer.writerow([',PDCA1_std,', str(np.std(self.stats_pdca1))])
            #     writer.writerow([',PDCA2_avg,', str(np.mean(self.stats_pdca2))])
            #     writer.writerow([',PDCA2_std,', str(np.std(self.stats_pdca2))])

            #     dist_pref_dir = [0] * self.nb_dupl_criteria
            #     for i in range(self.nb_models):
            #         if sum(self.nb_under_lim_prof_val[i].values()) > 0:
            #             dist_pref_dir[sum(self.nb_under_lim_prof_val[i].values())-1] += 1
            #     writer.writerow([',PDCA_avg_details,', str([float(i)/self.nb_models for i in dist_pref_dir])])
            # print([(h[m],m) for h in self.pdca1 for m in self.l_known_pref_dirs])
            # print([sum(self.pdca1[i].values())//len(self.l_known_pref_dirs) for i in range(self.nb_models)])
            #import pdb; pdb.set_trace()
            if len(self.l_known_pref_dirs)!=0:
                #print([sum(self.pdca1[i].values())//len(self.l_known_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ])
                #print(self.pdca1[0],sum(self.pdca1[0].values()))
                #import pdb; pdb.set_trace()
                self.stats_pdca1 = np.mean([sum(self.pdca1[i].values())//len(self.l_known_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ])
                self.stats_pdca2 = np.mean([sum(self.pdca1[i].values())/len(self.l_known_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ])
                writer.writerow([',PDCA1_avg,', str(np.mean([sum(self.pdca1[i].values())//len(self.l_known_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ]))])
                writer.writerow([',PDCA1_std,', str(np.std([sum(self.pdca1[i].values())//len(self.l_known_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ]))])
                writer.writerow([',PDCA2_avg,', str(np.mean([sum(self.pdca1[i].values())/len(self.l_known_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ]))])
                writer.writerow([',PDCA2_std,', str(np.std([sum(self.pdca1[i].values())/len(self.l_known_pref_dirs) for i in range(self.nb_models) if self.pdca1[i] ]))])
            
            writer.writerow([',%nb_exec_finished,', str((self.nb_models-self.ca_tests_avg.count(None))/float(self.nb_models))])



    def report_plot_results_csv(self):

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/plot_results_meta_mrsort3-rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_known_pref_dirs), dt)
        with open(filename, "w") as tmp_newfile:
            writer = csv.writer(tmp_newfile, delimiter=" ")
            #f = open(filename, 'w')
            #writer = csv.writer(f, delimiter = " ")
    
            writer.writerow(['SUMMARY DETAILS LIST'])
            writer.writerow([',exec_time_avg,' , [round(i,2) for i in self.exec_time if i is not None]])
            writer.writerow([',CAv_avg,', [round(i,2) for i in self.ca_avg if i is not None]])
            writer.writerow([',CAg_avg,', [round(i,2) for i in self.ca_tests_avg if i is not None]])
            if len(self.l_known_pref_dirs)!=0:        
                writer.writerow([',PDCA1_avg,', [round(sum(self.pdca1[i].values())//len(self.l_known_pref_dirs),2)  for i in range(self.nb_models) if self.pdca1[i] ]])
                writer.writerow([',PDCA2_avg,', [round(sum(self.pdca1[i].values())/len(self.l_known_pref_dirs),2)  for i in range(self.nb_models) if self.pdca1[i] ]])


    # build the instance transformed (with duplication of elements of performance table.)
    def build_osomcda_instance_random(self):
        criteria = [f.id for f in self.dupl_model_criteria]
        #import pdb; pdb.set_trace()
        # nb_criteria = len(criteria)*2 if not self.l_dupl_criteria else self.nb_dupl_criteria
        nb_crits = len(criteria) 
        
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/osomcda_rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_known_pref_dirs), dt)

        with open(filename, "w") as tmp_newfile:
            out = csv.writer(tmp_newfile, delimiter=" ")
            
            out.writerow(["criterion,direction," + ("," * (nb_crits - 2))])

            for crit in self.model.criteria:
                out.writerow([crit.id + "," + str(1) + "," + ("," * (nb_crits - 2))])
            for i in range(self.nb_criteria):
                if i in self.l_known_pref_dirs:
                    out.writerow([list(self.model.criteria.values())[i].id + "d," + str(-1) + "," + ("," * (nb_crits - 2))])
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
                dupl_nrow = [str(pt_values.performances[list(self.model.criteria.values())[i].id]) for i in range(self.nb_criteria) if i in self.l_known_pref_dirs]
                if self.l_known_pref_dirs:
                    out.writerow(["pt" + pt_values.id + "," + ",".join(nrow) + "," + ",".join(dupl_nrow) + "," + self.aa[pt_values.id].category_id])
                else :
                    out.writerow(["pt" + pt_values.id + "," + ",".join(nrow)  + "," + self.aa[pt_values.id].category_id])
                #tmp_newfile.flush()
        #import pdb; pdb.set_trace()
        return filename


    def learning_process(self):
        self.generate_random_instance()
        self.run_mrsort_all_models(self.nb_models)
        self.report_plot_results_csv()
        self.build_osomcda_instance_random()





if __name__ == "__main__":
    DATADIR = os.getenv('DATADIR')
    #file : without preference direction to learn
    #code for learning MRsort parameters with MIP formulation and without preference directions.
    
    #nb_categories = 2 #fixed
    #nb_criteria = 4
    #nb_criteria_sp = 1
    #dir_criteria = [1]*nb_criteria # by default 
    #dir_criteria = [-1, -1, -1, -1, -1, -1, -1, 2] # 1: increasing, -1:decreasing, 2:single-peaked, -2:single-valley 
    #dir_criteria = [1,1,1,1] # fixed to 1 for all criteria
    #l_known_pref_dirs = [] # the indices of criteria with unknown pref. dir. Fixed to [] if all preference directions are known already
    #l_known_pref_dirs = [0]
    #nb_alternatives = 100
    #nb_alternatives = 2**(nb_criteria-len(l_known_pref_dirs)) * 6**len(l_known_pref_dirs)
    #noise = 0.1
    #l_dupl_criteria = sorted(random.sample(list(range(6)),1))

    #nb_models: anciennmement nseed  
    #nb_tests = 10000
    #nb_models = 10
    
    #version_mip = 2
    #noise = None
    #noise = 0
    #mip_nb_threads = 1 # number of threads per MIP execution
    #mip_timetype = 1 # 1: for CPU time, 0: for wall clock time
    #mip_timeout = 60 # in seconds
    

    #inst = RandMRSortMIPLearning(nb_alternatives, nb_categories, nb_criteria, dir_criteria, l_known_pref_dirs, nb_tests, nb_models, noise = noise, version_mip = version_mip, mip_nb_threads = mip_nb_threads, mip_timetype=mip_timetype, mip_timeout=mip_timeout)
    #inst.learning_process()
    
    #inst.learning_process()
#    inst = RandMRSortLearning(nb_alternatives, nb_categories, \
#                    nb_criteria, dir_criteria, l_dupl_criteria, \
#                    nb_tests, nb_models, \
#                    meta_l, meta_ll, meta_nb_models, noise=noise)
#    inst.learning_process()


    #####################
    #STEP BY STEP PROCESS
    #import pdb; pdb.set_trace()
    # Generating a instance with some requirements of an equity on alternatives categorization  
    #inst.generate_random_instance()
    # # Print some parameters of the model :
    # print(inst.model.lbda)
    # print(inst.model.profiles)
    # print(inst.model.criteria)

    # inst.report_stats_parameters_csv()

    # inst.num_model = 0
    # execution_time = inst.run_mrsort()
    # print(execution_time)

    # #aa_learned = inst.model2.get_assignments(inst.pt_dupl)    
    # ca_v = inst.eval_model_validation()
    # print(ca_v)
 
    # matrix = compute_confusion_matrix(inst.aa, inst.aa_learned, inst.model.categories)
    # print_confusion_matrix(matrix, inst.model.categories)
    
    # ao_tests,al_tests,ca_t,cag_t = inst.eval_model_test()
    # print(ca_t)
    # matrix = compute_confusion_matrix(ao_tests,al_tests,inst.model.categories)
    # print_confusion_matrix(matrix,inst.model.categories)

    # inst.compute_stats_model()
    # inst.report_stats_model_csv()
    # inst.report_summary_results_csv()
    # inst.report_plot_results_csv()

    #inst.run_mrsort_all_models()
    # inst.build_osomcda_instance_random()
    
    #inst.learning_process()
    
    #import pdb; pdb.set_trace()
    #model,pt,cipibm_dir = create_random_model_mrsort(100, 2, 10, random_directions = [1]*10+[-1]*0)
    #duplicated_crit=[0,1,2,3,4,5]
    #duplicated_crit = list(range(6))
    #import pdb; pdb.set_trace()
    




