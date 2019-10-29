import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
import datetime
#import random
import time
#from collections import OrderedDict
#from itertools import product
#from pymcda.learning.meta_mrsort3 import MetaMRSortPop3, MetaMRSortPop3AUC
#from pymcda.learning.meta_mrsortvc4 import MetaMRSortVCPop4
from pymcda.learning.meta_mrsortvc4 import MetaMRSortVCPop4MSJP
#from pymcda.learning.heur_mrsort_init_profiles import HeurMRSortInitProfiles
#from pymcda.learning.lp_mrsort_weights import LpMRSortWeights
#from pymcda.learning.lp_mrsort_veto_weights import LpMRSortVetoWeights
#from pymcda.learning.lp_mrsort_weights_auc import LpMRSortWeightsAUC
#from pymcda.learning.heur_mrsort_profiles4 import MetaMRSortProfiles4
#from pymcda.learning.heur_mrsort_profiles5 import MetaMRSortProfiles5
#from pymcda.learning.heur_mrsort_veto_profiles5 import MetaMRSortVetoProfiles5
#from pymcda.learning.lp_mrsort_mobius import LpMRSortMobius
#from pymcda.learning.heur_mrsort_profiles_choquet import MetaMRSortProfilesChoquet
#from pymcda.types import CriterionValue, CriteriaValues
#from pymcda.types import Alternatives, Criteria, Criterion, PerformanceTable
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
from pymcda.generate import generate_random_performance_table_msjp
from pymcda.generate import duplicate_performance_table_msjp
#from pymcda.utils import compute_winning_and_loosing_coalitions
from pymcda.utils import compute_confusion_matrix, print_confusion_matrix
#import pdb
#from copy import deepcopy
from collections import Counter

#DATADIR = os.getenv('DATADIR', '%s/python_workspace/pymcda-master/pymcda-data' % os.path.expanduser('~'))
#DATADIR = os.getenv('DATADIR', '%s/python_workspace/MRSort-jupyter-notebook' % os.path.expanduser('~'))
DATADIR = os.getenv('DATADIR')

# meta_mrsort = MetaMRSortVCPop4

class RandMRSortLearning():
    def __init__(self, nb_alternatives, nb_categories, \
                    nb_criteria, dir_criteria, l_dupl_criteria, \
                    nb_tests, nb_models, \
                    meta_l, meta_ll, meta_nb_models, noise = None):
        self.nb_alternatives = nb_alternatives
        self.nb_categories = nb_categories
        self.nb_criteria = nb_criteria
        self.dir_criteria = dir_criteria
        self.nb_dupl_criteria = len(l_dupl_criteria)
        self.l_dupl_criteria = l_dupl_criteria
        self.nb_tests = nb_tests
        self.nb_models = nb_models
        self.meta_l = meta_l
        self.meta_ll = meta_ll
        self.meta_nb_models = meta_nb_models
        self.model = None
        self.pt = None
        self.noise = noise
        self.learned_models = []
        self.ca_avg = []
        self.ca_tests_avg = []
        self.ca_good_avg = 0
        self.ca_good_tests_avg = 0
        self.cpt_right_crit = dict()
        self.w_right_crit = [0]*(nb_models)
        self.w_dupl_right_crit = [0]*(nb_models)
        self.nb_null_weights = [0]*(nb_models)
        self.exec_time = [0]*(nb_models)
        self.cpt_gt_right_w = [0]*(nb_models) #count the number of time the good criteria got a greater weight than its duplicated criteria
        self.cpt_dupl_right_crit = dict()
        self.nb_under_lim_prof_val = [0]*(nb_models)
        self.nb_under_lim_prof_test = [0]*(nb_models)


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
                self.model = generate_random_mrsort_model_msjp(self.nb_criteria, self.nb_categories, random_directions = self.dir_criteria)
            if not ptafound:
                self.dupl_model_criteria = self.prepare_dupl_criteria_model()
                self.a = generate_alternatives(self.nb_alternatives)
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

    def run_mrsort(self):
        categories = self.model.categories_profiles.to_categories()
        if self.noise is None:
            meta = MetaMRSortVCPop4MSJP(self.meta_nb_models, self.dupl_model_criteria, categories, self.pt_dupl_sorted, self.aa)
        else:
            meta = MetaMRSortVCPop4MSJP(self.meta_nb_models, self.dupl_model_criteria, categories, self.pt_dupl_sorted, self.aa_noisy)
        
        t1 = time.time()
        for i in range(self.meta_l):
            self.model2, ca, all_models = meta.optimize(self.meta_ll,0)
            #print("%d: ca: %f" % (i, ca))
            if ca == 1:
                break
        t2 = time.time()
        self.exec_time[self.num_model] = (t2-t1)
        return self.exec_time[self.num_model]


    def eval_model_validation(self):
        self.aa_learned = self.model2.get_assignments(self.pt_dupl)  
        total = len(self.a)
        nok = 0
        totalg = 0
        okg = 0
        for alt in self.a:
            if self.aa(alt.id) != self.aa_learned(alt.id):
                nok += 1
            if self.aa(alt.id) == "cat1":
                totalg += 1
                if self.aa_learned(alt.id) == "cat1":
                    okg += 1
            # mesure the effect of enforcing constraints in the presence of duplicated criteria
            #import pdb; pdb.set_trace()
            for i,j in self.model2.criteria.items():
                if i[-1] == "d":
                    if self.model2.bpt['b1'].performances[i] > self.pt_dupl[alt.id].performances[i]:
                        self.nb_under_lim_prof_val[self.num_model] += 1
              
        totalg = 1 if totalg == 0 else totalg
        self.ca_avg += [(float(total-nok)/total)]
        self.ca_good_avg += (float(okg)/totalg)

        return (float(total-nok)/total),(float(okg)/totalg)


    def eval_model_test(self):
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
            for i,j in self.model2.criteria.items():
                if i[-1] == "d":
                    if self.model2.bpt['b1'].performances[i] > pt_tests_dupl[alt.id].performances[i]:
                        self.nb_under_lim_prof_test[self.num_model] += 1
            #import pdb; pdb.set_trace()
        totalg = 1 if totalg == 0 else totalg
        self.ca_tests_avg += [(float(total-nok)/total)]
        self.ca_good_tests_avg += (float(okg)/totalg)

        return ao_tests,al_tests,(float(total-nok)/total),(float(okg)/totalg)



    def run_mrsort_all_models(self):
        self.report_stats_parameters_csv()
        for m in range(self.nb_models):
            self.num_model = m
            self.run_mrsort()
            self.learned_models.append(self.model2)
            
            #evaluations            
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
        if self.dir_criteria is None:
            self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.nb_categories))  + "_dupl" + str(len(self.l_dupl_criteria)) + str_noise + "/"
        else:
            self.output_dir = "/rand_valid_test_na" + str(int(self.nb_alternatives)) + "_nca" + str(int(self.nb_categories)) + "_ncr" + str(int(self.dir_criteria.count(1))) + "-" + str(int(self.dir_criteria.count(-1))) + "_dupl" + str(len(self.l_dupl_criteria)) + str_noise + "/"

        if not os.path.exists(DATADIR + self.output_dir):
            os.mkdir(DATADIR + self.output_dir)

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/valid_test_dupl_meta_mrsort3-rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_dupl_criteria), dt)
        f = open(filename, 'w')
        self.writer = csv.writer(f, delimiter = " ")

        self.writer.writerow(['PARAMETERS'])

        self.writer.writerow([',nb_alernatives,', self.nb_alternatives])
        self.writer.writerow([',nb_categories,', self.nb_categories])
        self.writer.writerow([',nb_criteria,', self.nb_criteria])

        self.writer.writerow([',nb_outer_loops_meta,', self.meta_l])
        self.writer.writerow([',nb_inner_loops_meta,', self.meta_ll])
        self.writer.writerow([',nb_models_meta,', self.meta_nb_models])
        self.writer.writerow([',nb_learning_models,', self.nb_models])
        self.writer.writerow([',nb_alternatives_test,', self.nb_tests])

        self.writer.writerow(['ORIGINAL MODEL'])
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



    def report_stats_model_csv(self):
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
        self.writer.writerow([',assignments_id,', ",".join([str(i.id) for i in self.aa_learned])])
        self.writer.writerow([',assignments_cat,',",".join([str(i.category_id) for i in self.aa_learned])])

        self.writer.writerow(['LEARNED MODEL (validation)'])
        self.writer.writerow([',CA,', str(self.ca_avg[self.num_model])])
        if self.nb_dupl_criteria:
            self.writer.writerow([',%_alt_under_prof_val,', str(self.nb_under_lim_prof_val[self.num_model]/self.nb_alternatives/self.nb_dupl_criteria)])
        #self.writer.writerow([',CA_good,', str(cag_v)])

        self.writer.writerow(['MODEL TEST'])
        self.writer.writerow([',CA_tests,', str(self.ca_tests_avg[self.num_model])])
        if self.nb_dupl_criteria:
            self.writer.writerow([',%_alt_under_prof_test,', str(self.nb_under_lim_prof_test[self.num_model]/self.nb_tests/self.nb_dupl_criteria)])

        #self.writer.writerow([',CA_good_tests,', str(cag_t)])
        #import pdb; pdb.set_trace()
        


    def report_summary_results_csv(self):
        self.writer.writerow(['SUMMARY'])
        #writer.writerow(['%_right_crit_dir_avg,', str(float(sum(nb_right_crit_dir))/len(nb_right_crit_dir)/len(model.criteria))])
        #writer.writerow(['w_right_crit_dir_avg,', str(float(sum(w_right_crit_dir))/len(w_right_crit_dir))])
        #writer.writerow(['%_null_weights_avg,', str(float(sum(nb_null_weights))/len(nb_null_weights)/len(model.criteria))])
        self.writer.writerow([',comput_model_exec_time,' , str(float(sum(self.exec_time))/len(self.exec_time))])
        if self.l_dupl_criteria:
            self.writer.writerow([',%_dupl_right_crit_avg,', str(float(sum([i[1][0] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))]) # prob whenever we encounter a right criteria
            self.writer.writerow([',%_dupl_null_crit_avg,', str(float(sum([i[1][1] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))]) # prob whenever we encounter a null criteria
            self.writer.writerow([',%n_dupl_GODD_crit_avg,', str(float(sum([i[1][1]+i[1][0] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))]) # prob whenever we encounter a right criteria +null criteria = GOOD
        #import pdb; pdb.set_trace()
        self.writer.writerow([',%_dupl_GODD_crit_details_avg,', str([(j[0],round(float(j[1])/self.nb_models,3)) for j in dict(Counter([str(i[1][1]+i[1][0])+"cr" for i in self.cpt_dupl_right_crit.items()])).items()])]) # prob on the group of dupl criteria 

        if self.l_dupl_criteria:
            self.writer.writerow([',nb_dupl_wrong_crit_avg,', str(float(sum([i[1][2] for i in self.cpt_dupl_right_crit.items()]))/len(self.l_dupl_criteria)/len(self.cpt_dupl_right_crit))])
        self.writer.writerow([',w_dupl_right_weights_avg,', str(float(sum(self.w_dupl_right_crit))/len(self.w_dupl_right_crit))])
        self.writer.writerow([',nb_right_crit_avg,', str(float(sum([i[1][0] for i in self.cpt_right_crit.items()]))/len(self.cpt_right_crit))])
        self.writer.writerow([',nb_null_crit_avg,', str(float(sum([i[1][1] for i in self.cpt_right_crit.items()]))/len(self.cpt_right_crit))])
        if self.nb_dupl_criteria:
            self.writer.writerow([',%_w_right_greater_dupl,', str(float(sum(self.cpt_gt_right_w))/self.nb_models/self.nb_dupl_criteria)])
        self.writer.writerow([',w_right_weights_avg,', str(float(sum(self.w_right_crit))/len(self.w_right_crit))])
        self.writer.writerow([',CA_avg,', str(sum(self.ca_avg)/self.nb_models)])
        if self.nb_dupl_criteria:
            self.writer.writerow([',%_alt_under_prof_val_avg,', str(sum(self.nb_under_lim_prof_val)/self.nb_models/self.nb_alternatives/self.nb_dupl_criteria)])

        #self.writer.writerow([',CA_good_avg,', str(self.ca_good_avg/self.nb_models)])
        self.writer.writerow([',CA_tests_avg,', str(sum(self.ca_tests_avg)/self.nb_models)])
        if self.nb_dupl_criteria:
            self.writer.writerow([',%_alt_under_prof_val_test,', str(sum(self.nb_under_lim_prof_test)/self.nb_models/self.nb_tests/self.nb_dupl_criteria)])
        #self.writer.writerow([',CA_good_tests_avg,', str(self.ca_good_tests_avg/self.nb_models)])
        
        
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
        if self.l_dupl_criteria:
            writer.writerow([',n_dupl_GODD_crit_avg_list,', [round(float(i[1][1]+i[1][0])/len(self.l_dupl_criteria),2) for i in self.cpt_dupl_right_crit.items()]])
            #writer.writerow([',n_dupl_GODD_crit_avg_list,', [round(float(i)/len(self.l_dupl_criteria),2) for i in self.cpt_gt_right_w]])
        else:
            writer.writerow([',n_dupl_GODD_crit_avg_list,', [0]*self.nb_models])



    # build the instance transformed (with duplication of elements of performance table.)
    def build_osomcda_instance_random(self):
        criteria = [f.id for f in self.dupl_model_criteria]
        #import pdb; pdb.set_trace()
        # nb_criteria = len(criteria)*2 if not self.l_dupl_criteria else self.nb_dupl_criteria
        nb_crits = len(criteria) 
        
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "%s/osomcda_rand-%d-%d-%d-%d-%s.csv" \
                                % (DATADIR + self.output_dir,self.nb_alternatives, self.nb_categories, self.nb_criteria, len(self.l_dupl_criteria), dt)

        with open(filename, "w") as tmp_newfile:
            out = csv.writer(tmp_newfile, delimiter=" ")
            
            out.writerow(["criterion,direction," + ("," * (nb_crits - 2))])

            for crit in self.model.criteria:
                out.writerow([crit.id + "," + str(1) + "," + ("," * (nb_crits - 2))])
            for i in range(self.nb_criteria):
                if i in self.l_dupl_criteria:
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
                dupl_nrow = [str(pt_values.performances[list(self.model.criteria.values())[i].id]) for i in range(self.nb_criteria) if i in self.l_dupl_criteria]
                if self.l_dupl_criteria:
                    out.writerow(["pt" + pt_values.id + "," + ",".join(nrow) + "," + ",".join(dupl_nrow) + "," + self.aa[pt_values.id].category_id])
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
#    import time
#    import random
#    from pymcda.generate import generate_alternatives
#    from pymcda.generate import generate_random_performance_table
#    from pymcda.generate import generate_random_criteria_weights
#    from pymcda.generate import generate_random_mrsort_model_with_coalition_veto
#    from pymcda.generate import generate_random_mrsort_model
#    from pymcda.generate import generate_random_mrsort_model_msjp
#    from pymcda.utils import compute_winning_and_loosing_coalitions
#    from pymcda.utils import compute_confusion_matrix, print_confusion_matrix
#    from pymcda.types import AlternativePerformances
    #from pymcda.ui.graphic import display_electre_tri_models
    # input_file = "/Users/pegdwendeminoungou/python_workspace/pymcda-master/apps"

    nb_categories = 2 #fixed
    nb_criteria = 5
    nb_alternatives = 100
    dir_criteria = [1]*5 # fixed to 1 for all criteria
    l_dupl_criteria = list(range(nb_criteria))[:1]
    #l_dupl_criteria = sorted(random.sample(list(range(6)),1))

    nb_tests = 10000
    nb_models = 10

    #Parameters of the metaheuristic MRSort
    meta_l = 10
    meta_ll = 10
    meta_nb_models = 10

    #inst = RandMRSortLearning(nb_alternatives, nb_categories, \
    #                nb_criteria, dir_criteria, l_dupl_criteria, \
    #                nb_tests, nb_models, \
    #                meta_l, meta_ll, meta_nb_models)
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





