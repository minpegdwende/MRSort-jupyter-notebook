from __future__ import division
import errno
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
import random
import math
import copy
import time
import numpy as np
#from itertools import product
#from multiprocessing import Pool, Process, Queue
from pymcda.types import Criteria, Criterion
from multiprocessing import Process, Queue
#from threading import Thread

from pymcda.electre_tri import MRSort
#from pymcda.types import AlternativeAssignment, AlternativesAssignments
#from pymcda.types import PerformanceTable
#from pymcda.learning.heur_mrsort_init_veto_profiles import HeurMRSortInitVetoProfiles
from pymcda.learning.lp_mrsort_weights_impl8 import LpMRSortWeightsPositive
from pymcda.learning.lp_mrsort_veto_weights import LpMRSortVetoWeights
from pymcda.learning.heur_mrsort_profiles5_impl8 import MetaMRSortProfiles5
from pymcda.learning.heur_mrsort_veto_profiles5 import MetaMRSortVetoProfiles5
#from pymcda.utils import compute_ca
from pymcda.pt_sorted import SortedPerformanceTable
#from pymcda.generate import generate_random_mrsort_model
from pymcda.generate import generate_random_mrsort_model_with_coalition_veto
from pymcda.generate import generate_random_veto_profiles
from pymcda.generate import generate_alternatives
from pymcda.generate import generate_categories_profiles
from pymcda.generate import generate_random_profiles
from pymcda.generate import generate_random_profiles_msjp

def queue_get_retry(queue):
    while True:
        try:
            return queue.get()
        except IOError as e:
            if e.errno == errno.EINTR:
                continue
            else:
                raise

class MetaMRSortVCPop4():

    def __init__(self, nmodels, criteria, categories, pt_sorted, aa_ori,
                 lp_weights = LpMRSortWeightsPositive,
                 heur_profiles = MetaMRSortProfiles5,
                 lp_veto_weights = LpMRSortVetoWeights,
                 heur_veto_profiles= MetaMRSortVetoProfiles5,
                 seed = 0):
        self.nmodels = nmodels
        self.criteria = criteria
        self.categories = categories
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori

        self.lp_weights = lp_weights
        self.heur_profiles = heur_profiles
        self.lp_veto_weights = lp_veto_weights
        self.heur_veto_profiles = heur_veto_profiles

        self.metas = list()
        for i in range(self.nmodels):
            meta = self.init_one_meta(i + seed)
            self.metas.append(meta)

    def init_one_meta(self, seed):
        cps = generate_categories_profiles(self.categories)
        model = MRSort(self.criteria, None, None, None, cps)
        model.id = 'model_%d' % seed
        meta = MetaMRSortCV4(model, self.pt_sorted, self.aa_ori,
                             self.lp_weights,
                             self.heur_profiles,
                             self.lp_veto_weights,
                             self.heur_veto_profiles)
        random.seed(seed)
        meta.random_state = random.getstate()
        meta.auc = meta.model.auc(self.aa_ori, self.pt_sorted.pt)
        return meta

    def sort_models(self, fct_ca=0):
        # metas_sorted = sorted(self.metas, key = lambda (k): k.ca,
        #                       reverse = True)
        if fct_ca == 1:
            metas_sorted = sorted(self.metas, key = lambda k: k.ca_good,
                              reverse = True)
        elif fct_ca == 2:
            metas_sorted = sorted(self.metas, key = lambda k: k.ca_good + k.ca,
                              reverse = True)
        elif fct_ca ==3:
            metas_sorted = sorted(self.metas, key = lambda k: 1000*k.ca_good + k.ca,
                              reverse = True)
        else:
            metas_sorted = sorted(self.metas, key = lambda k: k.ca,
                              reverse = True)
        return metas_sorted

    def reinit_worst_models(self):
        metas_sorted = self.sort_models()
        nmeta_to_reinit = int(math.ceil(self.nmodels / 2))
        for meta in metas_sorted[nmeta_to_reinit:]:
            meta.init_profiles()

    def _process_optimize(self, meta, nmeta):
        random.setstate(meta.random_state)
        ca,ca_good = meta.optimize(nmeta)
        #print(ca,(meta.meta.good / meta.meta.na)*1.0,ca_good,(meta.meta.good_good / meta.meta.na_good)*1.0)
        #ca_good = meta.optimize(nmeta)
        # meta.queue.put([ca_good, meta.model.bpt, meta.model.cv, meta.model.lbda,
        #                 meta.model.vpt, meta.model.veto_weights,
        #                 meta.model.veto_lbda, random.getstate()])
        meta.queue.put([ca, ca_good, meta.model.bpt, meta.model.cv, meta.model.lbda,
                        meta.model.vpt, meta.model.veto_weights,
                        meta.model.veto_lbda, random.getstate()])

    def optimize(self, nmeta, fct_ca):
        self.reinit_worst_models()

        for meta in self.metas:
            meta.queue = Queue()
            meta.p = Process(target = self._process_optimize,
                             args = (meta, nmeta))
            meta.p.start()

        for meta in self.metas:
            output = queue_get_retry(meta.queue)

            meta.ca = output[0]
            meta.ca_good = output[1]
            meta.model.bpt = output[2]
            meta.model.cv = output[3]
            meta.model.lbda = output[4]
            meta.model.vpt = output[5]
            meta.model.veto_weights = output[6]
            meta.model.veto_lbda = output[7]
            meta.random_state = output[8]
            meta.auc = meta.model.auc(self.aa_ori, self.pt_sorted.pt)

#        self.models = {meta.model: meta.ca for meta in self.metas}

        #import pdb; pdb.set_trace()
        metas_sorted = self.sort_models(fct_ca)
        #import pdb; pdb.set_trace()
        #print(metas_sorted[0].ca, metas_sorted[0].ca_good)

        return metas_sorted[0].model, metas_sorted[0].ca, metas_sorted

class MetaMRSortCV4():

    def __init__(self, model, pt_sorted, aa_ori,
                 lp_weights = LpMRSortVetoWeights,
                 heur_profiles = MetaMRSortProfiles5,
                 lp_veto_weights = LpMRSortVetoWeights,
                 heur_veto_profiles = MetaMRSortVetoProfiles5):
        self.model = model
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori

        self.lp_weights = lp_weights
        self.heur_profiles = heur_profiles
        self.lp_veto_weights = lp_veto_weights
        self.heur_veto_profiles = heur_veto_profiles

        self.init_profiles()

        self.lp = self.lp_weights(self.model, self.pt_sorted.pt, self.aa_ori)
        self.lp.solve()
        self.meta = self.heur_profiles(self.model, self.pt_sorted, self.aa_ori)

        self.ca = self.meta.good / self.meta.na
        self.ca_good = 1 if (self.meta.na_good==0) else self.meta.good_good / self.meta.na_good

    def init_profiles(self):
        bpt = generate_random_profiles(self.model.profiles,
                                       self.model.criteria)
        self.model.bpt = bpt
        self.model.vpt = None

    def init_veto_profiles(self):
        worst = self.pt_sorted.pt.get_worst(self.model.criteria)
        vpt = generate_random_veto_profiles(self.model, worst)
        self.model.vpt = vpt

    def optimize(self, nmeta):
        self.lp.update_linear_program()
        self.lp.solve()
        self.meta.rebuild_tables()

        best_ca = self.meta.good / self.meta.na
        best_ca_good = 1 if (self.meta.na_good==0) else self.meta.good_good / self.meta.na_good
        best_bpt = self.model.bpt.copy()

        for i in range(nmeta):
            ca,ca_good = self.meta.optimize()
            if ca > best_ca:
                best_ca = ca
                best_ca_good = ca_good
                best_bpt = self.model.bpt.copy()

            if ca == 1:
                break

        self.model.bpt = best_bpt
        #print(best_ca,best_ca_good)

        # UNCOMMENT BELOW IF VETO IS TAKEN UNDER CONSIDERATION
#         if self.model.vpt is None:
#             self.init_veto_profiles()
#             best_vpt = None
#         else:
#             best_vpt = self.model.vpt.copy()

#         self.vlp = self.lp_veto_weights(self.model, self.pt_sorted.pt, self.aa_ori)
#         self.vlp.solve()

#         self.vmeta = self.heur_veto_profiles(self.model, self.pt_sorted,
#                                                  self.aa_ori)

# #        self.vlp.update_linear_program()
# #        self.vlp.solve()
#         self.vmeta.rebuild_tables()

#         best_ca = self.vmeta.good / self.vmeta.na
#         best_ca_good = self.vmeta.good_good / self.vmeta.na_good

#         for i in range(nmeta):
#             ca,ca_good = self.vmeta.optimize()
#             if ca > best_ca:
#                 best_ca = ca
#                 best_ca_good = ca_good
#                 best_vpt = self.model.vpt.copy()

#             if ca == 1:
#                 break

#         self.model.vpt = best_vpt

        return best_ca,best_ca_good




##############################################################################
##############################################################################
##############################################################################
#MSJP

# different because it uses MetaMRSortCV4MSJP (more randomized generation)
class MetaMRSortVCPop4MSJP():

    def __init__(self, nmodels, criteria, unk_pref_dir_criteria, categories, pt_sorted, aa_ori,
                 lp_weights = LpMRSortWeightsPositive, heur_profiles = MetaMRSortProfiles5,
                 lp_veto_weights = LpMRSortVetoWeights,heur_veto_profiles= MetaMRSortVetoProfiles5,
                 seed = 0, gamma = 0.5, renewal_method = 2, pretreatment_crit = None,duplication = True, 
                 fct_w_threshold = [0]*30, fct_percentile = {"c1":[(0,1)]*30}, renewal_models = (0.5,0), 
                 decision_rule = 1):
        self.nmodels = nmodels
        self.criteria = copy.deepcopy(criteria)
        self.nb_unk_criteria = len(unk_pref_dir_criteria)
        self.categories = categories
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori
        # finding p_th and (1-o)_th percentiles
        # self.prof_thresholds = prof_thresholds
        # self.w_threshold = w_threshold
        self.fct_percentile = fct_percentile
        self.fct_w_threshold = fct_w_threshold
        self.decision_rule = decision_rule
        self.renewal_models = renewal_models
        self.it_meta_max = 10 

        self.lp_weights = lp_weights
        self.heur_profiles = heur_profiles
        self.lp_veto_weights = lp_veto_weights
        self.heur_veto_profiles = heur_veto_profiles
        self.renewal_method = renewal_method
        self.gamma = gamma
        self.pretreatment_crit = pretreatment_crit
        self.duplication = duplication
        self.metas_cloned = [None] * int(math.ceil(self.nmodels / 2))

        self.metas = list()
        #random.seed()
        for i in range(self.nmodels):
            meta = self.init_one_meta(i + seed)
            meta.num = i
            meta.cah = 0
            self.metas.append(meta)
#        metas_sorted = sorted(self.metas, key = lambda k: k.ca,reverse = True)
#        for i in range(int(math.ceil(self.nmodels / 2))):
#            #import pdb; pdb.set_trace()
#            #cps = generate_categories_profiles(self.categories)
#            #model = MRSort(copy.deepcopy(self.criteria), None, None, None, cps)
#            self.metas_cloned[i] = metas_sorted[i].clone()
#            #print(self.metas_cloned[i],metas_sorted[i])
#            #import pdb; pdb.set_trace()
#        #print([m.ca for m in self.metas_cloned])
#        #print([m.ca for m in self.metas])


    def init_one_meta(self, seed):
        cps = generate_categories_profiles(self.categories)
        if not self.duplication and self.renewal_method > 0:
            for crit in self.criteria.values():
                if int(crit.id[1:]) <= self.nb_unk_criteria:
                    tmp = random.random()
                    if tmp < 0.5:
                        crit.direction = -1
                    else:
                        crit.direction = 1
                    #print(tmp)
            #print(self.unk_pref_dir_criteria,int(crit.id[1:]),self.criteria)
            #import pdb; pdb.set_trace()
        #print(self.criteria,self.renewal_method)
        #import pdb; pdb.set_trace()
        model = MRSort(copy.deepcopy(self.criteria), None, None, None, cps)
        model.id = 'model_%d' % seed
        meta = MetaMRSortCV4MSJP(model, self.pt_sorted, self.aa_ori, self.lp_weights, self.heur_profiles,
                             self.lp_veto_weights, self.heur_veto_profiles, gamma = self.gamma,
                             renewal_method = self.renewal_method, pretreatment_crit = self.pretreatment_crit,
                             duplication = self.duplication, nb_unk_criteria = self.nb_unk_criteria)
        random.seed(seed)
        meta.random_state = random.getstate()
        meta.auc = meta.model.auc(self.aa_ori, self.pt_sorted.pt)
        return meta

    def sort_models(self, fct_ca=0, heuristic = False):
        cps = generate_categories_profiles(self.categories)
        # metas_sorted = sorted(self.metas, key = lambda (k): k.ca,
        #                       reverse = True)
        if not heuristic:
            if fct_ca == 1:
                metas_sorted = sorted(self.metas, key = lambda k: k.ca_good,
                                  reverse = True)
            elif fct_ca == 2:
                metas_sorted = sorted(self.metas, key = lambda k: k.ca_good + k.ca,
                                  reverse = True)
            elif fct_ca ==3:
                metas_sorted = sorted(self.metas, key = lambda k: 1000*k.ca_good + k.ca,
                                  reverse = True)
            else:
                metas_sorted = sorted(self.metas, key = lambda k: k.ca,
                                  reverse = True)
        else:
            for m in self.metas:
                
                #print(self.pt_sorted)
                modelh = MRSort(copy.deepcopy(m.model.criteria), copy.deepcopy(m.model.cv),copy.deepcopy(m.model.bpt), copy.deepcopy(m.model.lbda), cps)
                #self.clone_model(self.modelh,self.model)
                #import pdb; pdb.set_trace()
                
                wtotal = 1-modelh.cv["c1"].value
                modelh.cv["c1"].value = 0
                #print(wtotal, modelh.cv)
#                if wtotal != 0:
#                    for el in modelh.cv:
#                        el.value /= wtotal
                modelh.bpt['b1'].performances["c1"] = 0
                
#                print(modelh.criteria,self.pt_sorted.pt)
                aa_learned = modelh.get_assignments(self.pt_sorted.pt)
                cah = 0
                for alt in self.aa_ori:
                    #print(alt)
                    #import pdb; pdb.set_trace()
                    if alt.category_id == aa_learned(alt.id):
                        cah += 1
                m.cah = cah / m.meta.na
                #if m.num == 0:
                #    print(wtotal, modelh.cv,modelh.lbda)
            #import pdb; pdb.set_trace()
            metas_sorted = sorted(self.metas, key = lambda k: ((k.ca)-k.cah), reverse = True)
            
        return metas_sorted


    def reinit_worst_models(self, cloning = False):
        #metas_sorted = self.sort_models()
        # Varying renewal rate =nmeta_to_reinit = nb model to reinitiate
        # renewal_models: (renewal_rate=0,renewal_coef_models=0)
        if self.renewal_models[0] == 0 and self.renewal_models[1] != 0:
            rr = round((1-np.mean([h.ca for h in self.metas]))/self.renewal_models[1],2)
            nmeta_to_reinit = int(math.ceil(self.nmodels * rr))
            #print(nmeta_to_reinit,(1-np.mean([h.ca for h in self.metas])), self.renewal_models[1],rr)
        #nmeta_to_reinit = int(math.ceil((1-np.mean([h.ca for h in self.metas]))/0.015))
        else:
            nmeta_to_reinit = int(math.ceil(self.nmodels * self.renewal_models[0]))
        if nmeta_to_reinit>self.nmodels:
            nmeta_to_reinit = self.nmodels
        
        if cloning:
            #update the best models
            #print([m.ca for m in self.metas])
            
            metas_sorted = sorted(self.metas + self.metas_cloned, key = lambda k: k.ca,reverse = True)
            self.metas = metas_sorted[:self.nmodels]
            #print([m.ca for m in self.metas_cloned])
            #print([m.ca for m in self.metas])
            #import pdb; pdb.set_trace()
            
            #cloning
            for i in range(nmeta_to_reinit):
                #self.metas[i] = metas_sorted[i]
                self.metas_cloned[i] = metas_sorted[i].clone()
                #print(self.metas_cloned[i],metas_sorted[i])
                #import pdb; pdb.set_trace()
            #print("new",[m.ca for m in self.metas_cloned])
            #print([m.ca for m in self.metas])
            #import pdb; pdb.set_trace()
        else :
            # heuristic true if modified ca into ca-cah
            metas_sorted = self.sort_models(heuristic = False)
        
        # add as parameters the percentage of -1/1 for each criteria
        perc_direction_renew = [0] * (self.nb_unk_criteria)
        #perc_direction_renew = [0.5] * (self.unk_pref_dir_criteria)
        if not self.duplication:
            for meta in metas_sorted[:self.nmodels-nmeta_to_reinit]:
                tmp_pdr = [x.direction for x in meta.model.criteria.values() if int(x.id[1:]) <= self.nb_unk_criteria]
                #print(tmp_pdr)
                perc_direction_renew = [perc_direction_renew[x] + (1 if tmp_pdr[x]==1 else 0) for x in range(len(tmp_pdr))]
            if self.nmodels-nmeta_to_reinit !=0:
                perc_direction_renew = [x/(self.nmodels - nmeta_to_reinit) for x in perc_direction_renew]
        if self.renewal_method == 1:
            perc_direction_renew = [0.5] * (self.nb_unk_criteria)

        #print(nmeta_to_reinit,perc_direction_renew)
        #import pdb; pdb.set_trace()
        if nmeta_to_reinit == self.nmodels:
            perc_direction_renew = [0.5] * (self.nb_unk_criteria)
        #print(perc_direction_renew)
        for meta in metas_sorted[self.nmodels - nmeta_to_reinit:]:
            meta.init_profiles(perc_direction_renew)
            #print([round(i,2) for i in meta.model.bpt['b1'].performances.values()])
        #print([x.model.criteria for x in metas_sorted])
        #import pdb; pdb.set_trace()

    def _process_optimize(self, meta, nmeta):
        random.setstate(meta.random_state)
        ca,ca_good = meta.optimize(nmeta)
        #print(ca,(meta.meta.good / meta.meta.na)*1.0,ca_good,(meta.meta.good_good / meta.meta.na_good)*1.0)
        #ca_good = meta.optimize(nmeta)
        # meta.queue.put([ca_good, meta.model.bpt, meta.model.cv, meta.model.lbda,
        #                 meta.model.vpt, meta.model.veto_weights,
        #                 meta.model.veto_lbda, random.getstate()])
        meta.queue.put([ca, ca_good, meta.model.bpt, meta.model.cv, meta.model.lbda,
                        meta.model.vpt, meta.model.veto_weights,
                        meta.model.veto_lbda, random.getstate()])

    def optimize(self, nmeta, fct_ca, it_meta=1, cloning = False):
        #print("ca",[(m.ca,m.num) for m in self.metas])
        #print("cv",[([round(i.value,2) for i in list(m.model.cv.values())],m.num) for m in self.metas])
        #print("bpt",[([round(i,2) for i in m.model.bpt['b1'].performances.values()],m.num) for m in self.metas])
        #print([m.meta.good for m in self.metas])
        if it_meta > 0:
            self.reinit_worst_models(cloning = cloning)
#        elif cloning:
#            metas_sorted = self.sort_models()
#            nmeta_to_reinit = int(math.ceil(self.nmodels / 2))
#            #print(self.metas_cloned)
#            #self.metas_cloned = copy.deepcopy(list(metas_sorted[:nmeta_to_reinit]))
#            for elem in metas_sorted[:nmeta_to_reinit]:
#                #import pdb; pdb.set_trace()
#                self.metas_cloned.append(copy.deepcopy(elem))
        
#        for m in self.metas:
#            print(m.model.profiles)
#            #print((round(m.ca,20),m.num), ([round(i.value,20) for i in list(m.model.cv.values())], round(m.model.lbda,20),m.num),([round(i,20) for i in m.model.bpt['b1'].performances.values()],m.num),m.meta.good)
        #print("ca","cv","bpt","good")
        #print("ca",[(round(m.ca,2),m.num) for m in self.metas])
        #print("cv",[([round(i.value,2) for i in list(m.model.cv.values())], round(m.model.lbda,2),m.num) for m in self.metas])
        #print("bpt",[([round(i,2) for i in m.model.bpt['b1'].performances.values()],m.num) for m in self.metas])
        #print([m.meta.good for m in self.metas])
        #import pdb; pdb.set_trace()

        for meta in self.metas:
            meta.it_meta = it_meta
            meta.fct_percentile = [self.fct_percentile["c"+str(u)][it_meta] for u in range(1,self.nb_unk_criteria+1)]
            meta.fct_w_threshold = self.fct_w_threshold[it_meta]
            meta.queue = Queue()
            meta.p = Process(target = self._process_optimize, args = (meta, nmeta))
            meta.p.start()

        for meta in self.metas:
            output = queue_get_retry(meta.queue)

            meta.ca = output[0]
            meta.ca_good = output[1]
            meta.model.bpt = output[2]
            meta.model.cv = output[3]
            meta.model.lbda = output[4]
            meta.model.vpt = output[5]
            meta.model.veto_weights = output[6]
            meta.model.veto_lbda = output[7]
            meta.random_state = output[8]
            meta.auc = meta.model.auc(self.aa_ori, self.pt_sorted.pt)
            #print(meta.model.criteria)
            #import pdb; pdb.set_trace()
        
        
        #print("ca",[m.tmpca for m in self.metas])
        #print("cv",[m.tmpcv for m in self.metas])
        #print("bpt",[m.tmpbpt for m in self.metas])
        #print([m.tmpgood for m in self.metas])
        #import pdb; pdb.set_trace()

#        self.models = {meta.model: meta.ca for meta in self.metas}

        #import pdb; pdb.set_trace()
        if cloning:
            #self.metas_cloned = []
            #print([m.ca for m in self.metas_cloned])
            #print([m.ca for m in self.metas])
            #import pdb; pdb.set_trace()
            metas_sorted = sorted(self.metas + self.metas_cloned, key = lambda k: k.ca,reverse = True)
            metas_sorted = metas_sorted[:self.nmodels]
            #import pdb; pdb.set_trace()
        else:
            metas_sorted = self.sort_models(fct_ca)
        
        
        #print([(m.ca,m.num) for m in self.metas])
        #print(self.metas[0].model.lbda, self.metas[0].model.cv, self.metas[0].model.bpt['b1'])
        #print("cah",[(m.cah,m.num) for m in self.metas])
        #print("ca sort",[(m.ca,m.num) for m in metas_sorted])
        #import pdb; pdb.set_trace()
        
        #print("cv",[([round(i.value,2) for i in list(m.model.cv.values())],m.num) for m in self.metas])
        #print("bpt",[([round(i,2) for i in m.model.bpt['b1'].performances.values()],m.num) for m in self.metas])
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        #print(metas_sorted[0].ca, metas_sorted[0].ca_good)
        #print([i for i in self.metas[0].model.bpt['b1'].performances.values()])
        #return metas_sorted[0].model, metas_sorted[0].ca, self.metas
        #code for returning the best model among exequo models
        #(best_model_baseline,best_model_1,best_model_3,best_model_5,best_model_7,best_model_9)
        #best_model = self.best_model_decision_rule(metas_sorted,decision = "baseline")
        
        #only for version 32,et cie
        # in presence of multiple criteria with unknown preference directions, rules changes...
        best_models = self.best_model_decision_rule(metas_sorted,decision = "baseline")
        #print("dir ",[m.model.criteria["c1"].direction for m in metas_sorted])
        #print(best_models)
        index = self.decision_rule
        return metas_sorted[best_models[index]].model, metas_sorted[best_models[index]].ca, metas_sorted, [metas_sorted[i] for i in best_models]
        
        #return metas_sorted[0].model, metas_sorted[0].ca, self.metas
    

    def best_model_decision_rule(self, models, decision = "baseline"):
        #code for returning the best model among exequo models
        # if ultim exquo, it chooses the most frequent up to 9
        # it return a list of best models according to different decision rules
        ca_max = models[0].ca
        #compteur pour specialement la baseline
        cpt = dict()
        for u in range(self.nb_unk_criteria):
            cpt["c"+str(u+1)] = [1,0,0,-1,0,0]
            i = 0
            while (i < len(models)) and (ca_max == models[i].ca):
                if models[i].model.criteria["c"+str(u+1)].direction == 1:
                    cpt["c"+str(u+1)][1] += 1
                    cpt["c"+str(u+1)][2] = i
                else:
                    cpt["c"+str(u+1)][4] += 1
                    cpt["c"+str(u+1)][5] = i
                i += 1
            if cpt["c"+str(u+1)][1] == cpt["c"+str(u+1)][4] and (i < len(models)):
                while (i < len(models)) and ((ca_max == models[i].ca) or (ca_max != models[i].ca and cpt["c"+str(u+1)][1] == cpt["c"+str(u+1)][4])):
                    if ca_max != models[i].ca:
                        ca_max = models[i].ca
                    if models[i].model.criteria["c"+str(u+1)].direction == 1:
                        cpt["c"+str(u+1)][1] += 1
                    else:
                        cpt["c"+str(u+1)][4] += 1
                    i += 1
        
        #compteur pour les autres qui garde en memoire les meilleurs models 1/-1
        cpt2 = dict()
        cpt3 = dict()
        for u in range(self.nb_unk_criteria):
            cpt2["c"+str(u+1)] = [1,0,0,-1,0,0]
            #cpt2[1] = len([i for i in range(len(models)) if models[i].model.criteria["c1"].direction == 1])
            if [i for i in range(len(models)) if models[i].model.criteria["c"+str(u+1)].direction == 1]:
                cpt2["c"+str(u+1)][2] = [i for i in range(len(models)) if models[i].model.criteria["c"+str(u+1)].direction == 1][0]
            #cpt2[4] = len([i for i in range(len(models)) if models[i].model.criteria["c1"].direction == -1])
            if [i for i in range(len(models)) if models[i].model.criteria["c"+str(u+1)].direction == -1]:
                cpt2["c"+str(u+1)][5] = [i for i in range(len(models)) if models[i].model.criteria["c"+str(u+1)].direction == -1][0]
            cpt3["c"+str(u+1)] = [m.model.criteria["c"+str(u+1)].direction for m in models]


        # 4 others decision rules based on majority of preference direction
        #best_model_1
        # in presence of mutiple criteria with unknown preference direction,
        # it can happen that one rule valids 1 model on one criteria, and another rule valids the same model with other criteria
        # not only this is not the case, but in presence of pair number of preference direction it is not possible to operate a 
        # majority rule (majority of models that are chosen accordingly to criteria with unknown preference directions)
        #if decision == "baseline":
        
        maxi = 0
        baseline = None
        best_model_baseline = []
        for u in range(self.nb_unk_criteria):
            if cpt["c"+str(u+1)][1] > cpt["c"+str(u+1)][4]:
                best_model_baseline += [1]
            elif cpt["c"+str(u+1)][4] > cpt["c"+str(u+1)][1]:
                best_model_baseline += [-1]
            elif cpt3["c"+str(u+1)].count(1) > (len(models)/2)-1 :
                best_model_baseline += [1]
            else:
                best_model_baseline += [-1]
            maxi = max(maxi,cpt["c"+str(u+1)][2],cpt["c"+str(u+1)][5])
        for m in range(len(models)):
            if m <= maxi:
                if best_model_baseline == [models[m].model.criteria["c"+str(u+1)].direction for u in range(self.nb_unk_criteria)]:
                    baseline = m
                    break
        if not baseline:
            baseline = 0
        
        #1er model quoi qu'il en soit
        model_1 = 0

        #return best_model_1
#    #elif decision == "3":
#        if cpt3[:3].count(1) > 1:
#            best_model_3 = cpt2[2]
#        else:
#            best_model_3 = cpt2[5]
#        #return best_model_3
#    #elif decision == "5":
#        if cpt3[:5].count(1) > 2:
#            best_model_5 = cpt2[2]
#        else:
#            best_model_5 = cpt2[5]
#        #return best_model_5
#    #elif decision == "7":
#        if cpt3[:7].count(1) > 3:
#            best_model_7 = cpt2[2]
#        else:
#            best_model_7 = cpt2[5]
        #return best_model_7
    #elif decision == "9":
        #le model rendu par la majorité de models
        maxi = 0
        best_model_last = []
        model_last = None
        ind = (len(models)-1) if (len(models)%2==0) else (len(models)-2)
        for u in range(self.nb_unk_criteria):
            if cpt3["c"+str(u+1)][:ind].count(1) >= len(models)/2:
                best_model_last += [1]
            else:
                best_model_last += [-1]
            maxi = max(maxi,cpt2["c"+str(u+1)][2],cpt2["c"+str(u+1)][5])
        for m in range(len(models)):
            if m <= maxi:
                if best_model_last == [models[m].model.criteria["c"+str(u+1)].direction for u in range(self.nb_unk_criteria)]:
                    model_last = m
                    break
        if not model_last:
            model_last = 0
        #return best_model_9
        #print(best_model_baseline,best_model_1,best_model_3,best_model_5,best_model_7,best_model_9)
        #return (model_baseline,best_model_1,best_model_3,best_model_5,best_model_7,best_model_9)
        return (baseline,model_1,model_last)


# different from MetaMRSortCV4 because it uses generate_random_profiles_msjp (truely random)
class MetaMRSortCV4MSJP():

    def __init__(self, model, pt_sorted, aa_ori, lp_weights = LpMRSortVetoWeights, 
                 heur_profiles = MetaMRSortProfiles5, lp_veto_weights = LpMRSortVetoWeights,
                 heur_veto_profiles = MetaMRSortVetoProfiles5, gamma = 0.5, renewal_method = 2,
                 pretreatment_crit = None, duplication = False, fct_percentile = [(0,1)], 
                 fct_w_threshold = 0, nb_unk_criteria = 0):
        self.model = model
        self.modelh = None
        self.pt_sorted = pt_sorted
        self.aa_ori = aa_ori
        self.nb_unk_criteria = nb_unk_criteria
        #self.tmpca = None
        #self.tmpcv = None
        #self.tmpbpt = None
        #self.tmpgood = None

        self.lp_weights = lp_weights
        self.heur_profiles = heur_profiles
        self.lp_veto_weights = lp_veto_weights
        self.heur_veto_profiles = heur_veto_profiles
        self.duplication = duplication
        self.renewal_method = renewal_method
        self.gamma = gamma
        self.pretreatment_crit = pretreatment_crit
        self.it_meta = 0
        self.fct_w_threshold = fct_w_threshold
        self.fct_percentile = [(0,1)]*nb_unk_criteria

        self.init_profiles()
        #print(self.fct_percentile)
        #print(self.model.bpt['b1'].performances)
        self.lp = self.lp_weights(self.model, self.pt_sorted.pt, self.aa_ori, gamma = gamma, 
                                  renewal_method = renewal_method, pretreatment_crit = pretreatment_crit)
        self.lp.solve()
        self.meta = self.heur_profiles(self.model, self.pt_sorted, self.aa_ori)
        #print(self.model.bpt['b1'].performances)
        
        self.ca = self.meta.good / self.meta.na
        self.ca_good = 1 if (self.meta.na_good==0) else self.meta.good_good / self.meta.na_good
        

    def init_profiles(self, perc_direction_renew = []):
        #tmp_criteria = copy.deepcopy(self.model.criteria)
        i = 0
        if not self.duplication and self.renewal_method > 0: 
            for crit in self.model.criteria:
                if int(crit.id[1:]) <= len(perc_direction_renew):
                    tmp = random.random()
                    #print(tmp)
                    if self.renewal_method == 2:
                        crit.direction = 1 if tmp < perc_direction_renew[i] else -1
                        #print("renew ",crit)
                    if self.renewal_method == 3:
                        crit.direction = -1 if tmp < perc_direction_renew[i] else 1
                    i += 1
        #tmp = int(random.random()*100)
        #random.seed(tmp)
#        if self.it_meta > 20:
#            bpt = generate_random_profiles_msjp(self.model.profiles,self.model.criteria)
#        else:
        #A CHANGER in presence with multiple criteria with unknown preference directions
        bpt = generate_random_profiles_msjp(self.model.profiles,self.model.criteria, fct_percentile = self.fct_percentile, nb_unk_criteria = self.nb_unk_criteria)
        
        #print(int(crit.id[1:]),perc_direction_renew,self.model.criteria)
        #import pdb; pdb.set_trace()
        self.model.bpt = bpt
        #print(tmp,[round(i,2) for i in self.model.bpt['b1'].performances.values()])
        self.model.vpt = None

    def init_veto_profiles(self):
        worst = self.pt_sorted.pt.get_worst(self.model.criteria)
        vpt = generate_random_veto_profiles(self.model, worst)
        self.model.vpt = vpt

    def optimize(self, nmeta):
        #print(self.fct_w_threshold,self.fct_percentile)
        #print(self.model.bpt['b1'].performances)
        self.lp.update_linear_program(fct_w_threshold = self.fct_w_threshold, nb_unk_criteria = self.nb_unk_criteria)
        self.lp.solve()
        self.meta.rebuild_tables()
        #import pdb; pdb.set_trace()
#        self.tmpca = (round(self.ca,10),self.num)
#        self.tmpcv = [([round(i.value,10) for i in list(self.model.cv.values())],self.num)]
#        self.tmpbpt = [([round(i,6) for i in self.model.bpt['b1'].performances.values()],self.num)]
#        self.tmpgood = (self.meta.good, self.meta.good/self.meta.na,self.num)
#        #time.sleep(1)
#        print(self.tmpca,self.tmpcv,self.tmpbpt,self.tmpgood,round(self.model.lbda,10))
        #print("ca",[(m.ca,m.num) for m in self.metas])
        #print("cv",[([round(i.value,2) for i in list(m.model.cv.values())],m.num) for m in self.metas])
        #print("bpt",[([round(i,2) for i in m.model.bpt['b1'].performances.values()],m.num) for m in self.metas])
        #print([m.meta.good for m in self.metas])

        best_ca = self.meta.good / self.meta.na
        best_ca_good = 1 if (self.meta.na_good==0) else self.meta.good_good / self.meta.na_good
        best_bpt = self.model.bpt.copy()

        for i in range(nmeta):
            #print(self.prof_thresholds)
            ca,ca_good = self.meta.optimize(fct_percentile = self.fct_percentile, nb_unk_criteria = self.nb_unk_criteria)
            if ca > best_ca:
                best_ca = ca
                best_ca_good = ca_good
                best_bpt = self.model.bpt.copy()

            if ca == 1:
                break

        self.model.bpt = best_bpt
        
        #self.modelh = MRSort(self.model.criteria, self.model.cv, self.model.bpt, self.model.lbda, self.model.categories_profiles)
        #MRSort(c, cv, bpt, lbda, cps)
        #MetaMRSortCV4MSJP(self.model,self.pt_sorted, self.aa_ori, self.lp_weights, self.heur_profiles)
        #self.clone_model(self.modelh,self.model)
#        self.modelh.cv["c1"].value = 0
#        self.modelh.bpt['b1'].performances["c1"] = 0
#        aa_learned = self.modelh.get_assignments(self.pt_sorted)
#        cah = 0
#        for alt in self.aa_ori:
#            if alt.category_id == aa_learned(alt.id):
#                cah += 1
#        self.cah = cah / self.meta.na
        #print(self.aa_ori)
        #import pdb; pdb.set_trace()
        #print(best_ca,best_ca_good)

        # UNCOMMENT BELOW IF VETO IS TAKEN UNDER CONSIDERATION
#         if self.model.vpt is None:
#             self.init_veto_profiles()
#             best_vpt = None
#         else:
#             best_vpt = self.model.vpt.copy()

#         self.vlp = self.lp_veto_weights(self.model, self.pt_sorted.pt, self.aa_ori)
#         self.vlp.solve()

#         self.vmeta = self.heur_veto_profiles(self.model, self.pt_sorted,
#                                                  self.aa_ori)

# #        self.vlp.update_linear_program()
# #        self.vlp.solve()
#         self.vmeta.rebuild_tables()

#         best_ca = self.vmeta.good / self.vmeta.na
#         best_ca_good = self.vmeta.good_good / self.vmeta.na_good

#         for i in range(nmeta):
#             ca,ca_good = self.vmeta.optimize()
#             if ca > best_ca:
#                 best_ca = ca
#                 best_ca_good = ca_good
#                 best_vpt = self.model.vpt.copy()

#             if ca == 1:
#                 break

#         self.model.vpt = best_vpt

        return best_ca,best_ca_good

    def clone_model(self,mod1,mod2):
        mod1.criteria = copy.deepcopy(mod2.criteria)
        mod1.cv = copy.deepcopy(mod2.cv)
        mod1.bpt = copy.deepcopy(mod2.bpt)
        mod1.cprofiles = copy.deepcopy(mod2.cprofiles)
        mod1.categories = copy.deepcopy(mod2.categories)
        mod1.profiles = copy.deepcopy(mod2.profiles)
        mod1.vpt = copy.deepcopy(mod2.vpt)
        mod1.preference = copy.deepcopy(mod2.preference)
        mod1.indifference = copy.deepcopy(mod2.indifference)
        mod1.id = copy.deepcopy(mod2.id)
        mod1.veto_weights = copy.deepcopy(mod2.veto_weights)
        mod1.lbda = copy.deepcopy(mod2.lbda)
        mod1.veto_lbda = copy.deepcopy(mod2.veto_lbda)
        
#    def clone_lp_lp(self,lp1,lp2):
#        lp1._disposed = copy.deepcopy(lp2._disposed)
#        lp1._aborter = copy.deepcopy(lp2._aborter)
#        import pdb; pdb.set_trace()
#        lp1._env = copy.deepcopy(lp2._env)
#        lp1._lp = copy.deepcopy(lp2._lp)
#        lp1._pslst = copy.deepcopy(lp2._pslst)
#        import pdb; pdb.set_trace()
#        lp1._env_lp_ptr = copy.deepcopy(lp2._env_lp_ptr)
#        lp1.variables = copy.deepcopy(lp2.variables)
#        lp1.parameters = copy.deepcopy(lp2.parameters)
#        lp1.linear_constraints = copy.deepcopy(lp2.linear_constraints)
#        lp1.quadratic_constraints = copy.deepcopy(lp2.quadratic_constraints)
#        lp1.indicator_constraints = copy.deepcopy(lp2.indicator_constraints)
#        import pdb; pdb.set_trace()
#        lp1.SOS = copy.deepcopy(lp2.SOS)
#        lp1.objective = copy.deepcopy(lp2.objective)
#        lp1.multiobj = copy.deepcopy(lp2.multiobj)
#        lp1.MIP_starts = copy.deepcopy(lp2.MIP_starts)
#        lp1.solution = copy.deepcopy(lp2.solution)
#        lp1.presolve = copy.deepcopy(lp2.presolve)
#        lp1.order = copy.deepcopy(lp2.order)
#        lp1.conflict = copy.deepcopy(lp2.conflict)
#        lp1.advanced = copy.deepcopy(lp2.advanced)
#        lp1.start = copy.deepcopy(lp2.start)
#        lp1.feasopt = lp2.feasopt
#        lp1.long_annotations = copy.deepcopy(lp2.long_annotations)
#        lp1.double_annotations = copy.deepcopy(lp2.double_annotations)
#        lp1.pwl_constraints = copy.deepcopy(lp2.pwl_constraints)


    def clone(self):
        mnew = MetaMRSortCV4MSJP(self.model,self.pt_sorted, self.aa_ori, self.lp_weights, self.heur_profiles)
        #import pdb; pdb.set_trace()
        mnew.it_meta = self.it_meta
        mnew.auc = self.auc
        mnew.ca = self.ca
        mnew.num = self.num
        mnew.random_state = self.random_state
        #import pdb; pdb.set_trace()
        self.clone_model(mnew.model,self.model)
        #mnew.model.el = copy.deepcopy(getattr(other, "self.model."+el))
        #mnew.model = copy.deepcopy(self.model)
        
        #self.lp.__dict__.keys()
        #self.meta.__dict__.keys()
        #mnew.lp.renewal_method = self.lp.renewal_method
        #import pdb; pdb.set_trace()
        #self.clone_model(mnew.lp.model,self.lp.model)
        mnew.meta.na = copy.deepcopy(self.meta.na)
        mnew.meta.na_good = copy.deepcopy(self.meta.na_good)
        mnew.meta.nc = copy.deepcopy(self.meta.nc)
        #mnew.meta.fct_w_threshold = copy.deepcopy(self.meta.fct_w_threshold)
        self.clone_model(mnew.meta.model,self.meta.model)
        mnew.meta.nprofiles = copy.deepcopy(self.meta.nprofiles)
        mnew.meta.pt_sorted = self.meta.pt_sorted
        mnew.meta.aa_ori = self.meta.aa_ori
        mnew.meta.cat = copy.deepcopy(self.meta.cat)
        mnew.meta.cat_ranked = copy.deepcopy(self.meta.cat_ranked)
        mnew.meta.bp = copy.deepcopy(self.meta.bp)
        mnew.meta.b0 = copy.deepcopy(self.meta.b0)
        mnew.meta.ct = copy.deepcopy(self.meta.ct)
        mnew.meta.good = self.meta.good
        mnew.meta.good_good = self.meta.good_good
        mnew.meta.vt = copy.deepcopy(self.meta.vt)
        mnew.meta.aa = copy.deepcopy(self.meta.aa)
        
        mnew.lp = mnew.lp_weights(mnew.model, mnew.pt_sorted.pt, mnew.aa_ori, gamma = mnew.gamma, 
                                  renewal_method = mnew.renewal_method, pretreatment_crit = mnew.pretreatment_crit)
        mnew.lp.solve()
#        mnew.lp.delta = self.lp.delta
#        mnew.lp.gamma = self.lp.gamma
#        mnew.lp.w1_threshold = self.lp.w1_threshold
#        mnew.lp.w_threshold = self.lp.w_threshold
#        mnew.lp.it_meta = self.lp.it_meta
#        mnew.lp.pretreatment_crit = self.lp.pretreatment_crit
#        
#        mnew.lp.cat_ranks = copy.deepcopy(self.lp.cat_ranks)
#        mnew.lp.pt = copy.deepcopy(self.lp.pt)
#        mnew.lp.aa_ori = self.lp.aa_ori
#        mnew.lp.c_xi = copy.deepcopy(self.lp.c_xi)
#        mnew.lp.c_yi = copy.deepcopy(self.lp.c_yi)
#        mnew.lp.a_c_xi = copy.deepcopy(self.lp.a_c_xi)
#        mnew.lp.a_c_yi = copy.deepcopy(self.lp.a_c_yi)
#        import pdb; pdb.set_trace()
#        self.clone_lp_lp(mnew.lp.lp,self.lp.lp)
#        import pdb; pdb.set_trace()
#        mnew.lp.solve_function = self.lp.solve_function
        #import pdb; pdb.set_trace()
        #mnew.meta = copy.deepcopy(self.meta)
        #import pdb; pdb.set_trace()
        return mnew


if __name__ == "__main__":
    import time
    #import random
    #from pymcda.generate import generate_alternatives
    from pymcda.generate import generate_random_performance_table
    #from pymcda.generate import generate_random_criteria_weights
    #from pymcda.generate import generate_random_mrsort_model_with_coalition_veto
    #from pymcda.utils import compute_winning_and_loosing_coalitions
    from pymcda.utils import compute_confusion_matrix, print_confusion_matrix
    from pymcda.types import AlternativePerformances
    from pymcda.ui.graphic import display_electre_tri_models

    # Generate a random ELECTRE TRI BM model
    model = generate_random_mrsort_model_with_coalition_veto(7, 2, 5,
                                                             veto_weights = True)
#    model = generate_random_mrsort_model(7, 2, 1)
    worst = AlternativePerformances("worst",
                                     {c.id: 0 for c in model.criteria})
    best = AlternativePerformances("best",
                                    {c.id: 1 for c in model.criteria})

    # Generate a set of alternatives
    a = generate_alternatives(1000)
    pt = generate_random_performance_table(a, model.criteria)
    aa = model.get_assignments(pt)

    nmeta = 20
    nloops = 10

    print('Original model')
    print('==============')
    cids = model.criteria.keys()
    model.bpt.display(criterion_ids = cids)
    model.cv.display(criterion_ids = cids)
    print("lambda\t%.7s" % model.lbda)
    if model.vpt is not None:
        model.vpt.display(criterion_ids = cids)
    if model.veto_weights is not None:
        model.veto_weights.display(criterion_ids = cids)
        print("veto_lambda\t%.7s" % model.veto_lbda)

    ncriteria = len(model.criteria)
    ncategories = len(model.categories)
    pt_sorted = SortedPerformanceTable(pt)

    t1 = time.time()

    categories = model.categories_profiles.to_categories()
    meta = MetaMRSortVCPop4(10, model.criteria, categories, pt_sorted, aa)
    for i in range(nloops):
        model2, ca = meta.optimize(nmeta,0)
        print("%d: ca: %f" % (i, ca))
        if ca == 1:
            break

    t2 = time.time()
    print("Computation time: %g secs" % (t2-t1))

    print('Learned model')
    print('=============')
    model2.bpt.display(criterion_ids = cids)
    model2.cv.display(criterion_ids = cids)
    print("lambda\t%.7s" % model2.lbda)
    if model2.vpt is not None:
        model2.vpt.display(criterion_ids = cids)
    if model2.veto_weights is not None:
        model2.veto_weights.display(criterion_ids = cids)
        print("veto_lambda\t%.7s" % model2.veto_lbda)

    aa_learned = model2.get_assignments(pt)

    total = len(a)
    nok = 0
    anok = []
    for alt in a:
        if aa(alt.id) != aa_learned(alt.id):
            anok.append(alt)
            nok += 1

    print("Good assignments: %g %%" % (float(total-nok)/total*100))
    print("Bad assignments : %g %%" % (float(nok)/total*100))

    matrix = compute_confusion_matrix(aa, aa_learned, model.categories)
    print_confusion_matrix(matrix, model.categories)

    model.id = "original"
    model2.id = "learned"
    display_electre_tri_models([model, model2],
                               [worst, worst], [best, best],
                               [[ap for ap in model.vpt],
                                [ap for ap in model2.vpt]])
